/**
 * frontend/src/services/api.js
 *
 * Central API client for ZentraX AI. Wraps Axios with:
 *   - Base URL configuration (env-driven, works across dev/staging/prod).
 *   - Automatic Authorization header attachment.
 *   - Normalized error handling via `ApiError`.
 *   - A small set of modular, documented request functions for chat and
 *     user data, meant to be imported directly by components/pages
 *     (Sidebar, ChatBox, login page, etc.) rather than calling Axios
 *     directly from UI code.
 *
 * Environment:
 *   NEXT_PUBLIC_API_BASE_URL   e.g. https://api.zentrax.ai or http://localhost:8000
 *
 * Auth token storage:
 *   This module is storage-agnostic — it calls `getAuthToken()` /
 *   `setAuthToken()` / `clearAuthToken()`, which default to a thin
 *   localStorage wrapper. If ZentraX moves to httpOnly session cookies
 *   (recommended for XSS resistance), replace those three functions with
 *   no-ops and let the browser attach the cookie automatically — leave
 *   `withCredentials: true` on the Axios instance in that case.
 */

import axios from "axios";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000";
const AUTH_TOKEN_STORAGE_KEY = "zentrax_auth_token";
const REQUEST_TIMEOUT_MS = 20000;

// --------------------------------------------------------------------------- //
// Token storage (swap out for httpOnly cookies if/when the backend supports it)
// --------------------------------------------------------------------------- //

export function getAuthToken() {
  if (typeof window === "undefined") return null; // SSR guard
  return window.localStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
}

export function setAuthToken(token) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
}

export function clearAuthToken() {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(AUTH_TOKEN_STORAGE_KEY);
}

// --------------------------------------------------------------------------- //
// Normalized error type
// --------------------------------------------------------------------------- //

/**
 * Uniform error shape for all API failures, whether they come from the
 * server (4xx/5xx with a JSON body), the network (no response at all), or
 * client-side setup mistakes.
 */
export class ApiError extends Error {
  constructor({ message, status = null, code = null, details = null }) {
    super(message);
    this.name = "ApiError";
    this.status = status; // HTTP status code, or null for network errors
    this.code = code; // backend-defined error code, if provided
    this.details = details; // raw backend error payload, for debugging/forms
  }

  get isAuthError() {
    return this.status === 401 || this.status === 403;
  }

  get isNetworkError() {
    return this.status === null;
  }
}

function normalizeError(error) {
  if (axios.isCancel(error)) {
    return new ApiError({ message: "Request was cancelled", code: "CANCELLED" });
  }

  if (error.response) {
    const { status, data } = error.response;
    return new ApiError({
      message: data?.message || data?.detail || "The server returned an error.",
      status,
      code: data?.code || null,
      details: data || null,
    });
  }

  if (error.request) {
    return new ApiError({
      message: "Unable to reach the server. Check your connection and try again.",
      status: null,
      code: "NETWORK_ERROR",
    });
  }

  return new ApiError({ message: error.message || "An unexpected error occurred." });
}

// --------------------------------------------------------------------------- //
// Axios instance
// --------------------------------------------------------------------------- //

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: REQUEST_TIMEOUT_MS,
  headers: {
    "Content-Type": "application/json",
  },
});

apiClient.interceptors.request.use((config) => {
  const token = getAuthToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

let onUnauthorized = null;

/**
 * Register a callback invoked whenever a request comes back 401
 * (e.g. to clear local auth state and redirect to /login). Call this once
 * near app startup, e.g. in _app.js.
 */
export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler;
}

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError = normalizeError(error);
    if (apiError.status === 401) {
      clearAuthToken();
      onUnauthorized?.();
    }
    return Promise.reject(apiError);
  }
);

// --------------------------------------------------------------------------- //
// Chat endpoints
// --------------------------------------------------------------------------- //

/**
 * Send a chat message and receive the assistant's reply.
 *
 * @param {string} chatId - Target conversation id.
 * @param {string} message - The user's message text.
 * @param {{ signal?: AbortSignal }} [options] - Optional AbortController signal to cancel in-flight requests.
 * @returns {Promise<{ id: string, role: string, content: string, createdAt: string }>}
 * @throws {ApiError}
 */
export async function sendChatMessage(chatId, message, options = {}) {
  const { data } = await apiClient.post(
    `/api/chats/${encodeURIComponent(chatId)}/messages`,
    { message },
    { signal: options.signal }
  );
  return data;
}

/**
 * Fetch message history for a single chat.
 *
 * @param {string} chatId
 * @param {{ limit?: number, before?: string, signal?: AbortSignal }} [options]
 * @returns {Promise<Array<{ id: string, role: string, content: string, createdAt: string }>>}
 * @throws {ApiError}
 */
export async function fetchChatHistory(chatId, options = {}) {
  const { limit, before, signal } = options;
  const { data } = await apiClient.get(`/api/chats/${encodeURIComponent(chatId)}/messages`, {
    params: { limit, before },
    signal,
  });
  return data;
}

/**
 * Fetch the list of chats (sidebar conversation list) for the current user.
 *
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<Array<{ id: string, title: string, updatedAt: string }>>}
 * @throws {ApiError}
 */
export async function fetchChats(options = {}) {
  const { data } = await apiClient.get("/api/chats", { signal: options.signal });
  return data;
}

/**
 * Create a new, empty chat and return its id/metadata.
 *
 * @param {{ title?: string, signal?: AbortSignal }} [options]
 * @returns {Promise<{ id: string, title: string }>}
 * @throws {ApiError}
 */
export async function createChat(options = {}) {
  const { title, signal } = options;
  const { data } = await apiClient.post("/api/chats", { title }, { signal });
  return data;
}

/**
 * Permanently delete a chat and its message history.
 *
 * @param {string} chatId
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<void>}
 * @throws {ApiError}
 */
export async function deleteChat(chatId, options = {}) {
  await apiClient.delete(`/api/chats/${encodeURIComponent(chatId)}`, { signal: options.signal });
}

// --------------------------------------------------------------------------- //
// User endpoints
// --------------------------------------------------------------------------- //

/**
 * Fetch the current authenticated user's profile.
 *
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<{ id: string, name: string, email: string }>}
 * @throws {ApiError}
 */
export async function fetchCurrentUser(options = {}) {
  const { data } = await apiClient.get("/api/users/me", { signal: options.signal });
  return data;
}

/**
 * Update fields on the current user's profile.
 *
 * @param {Partial<{ name: string, email: string }>} updates
 * @param {{ signal?: AbortSignal }} [options]
 * @returns {Promise<{ id: string, name: string, email: string }>}
 * @throws {ApiError}
 */
export async function updateCurrentUser(updates, options = {}) {
  const { data } = await apiClient.patch("/api/users/me", updates, { signal: options.signal });
  return data;
}

// --------------------------------------------------------------------------- //
// Auth endpoints
// --------------------------------------------------------------------------- //

/**
 * Log in with email/password and persist the returned session token.
 *
 * @param {{ email: string, password: string }} credentials
 * @returns {Promise<{ token: string, user: { id: string, name: string, email: string } }>}
 * @throws {ApiError}
 */
export async function login({ email, password }) {
  const { data } = await apiClient.post("/api/auth/login", { email, password });
  if (data?.token) setAuthToken(data.token);
  return data;
}

/**
 * Log out the current session, both server-side and locally.
 * Local token/state is cleared even if the server call fails.
 *
 * @returns {Promise<void>}
 */
export async function logout() {
  try {
    await apiClient.post("/api/auth/logout");
  } finally {
    clearAuthToken();
  }
}