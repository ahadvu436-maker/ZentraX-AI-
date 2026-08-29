/**
 * frontend/src/store/useAppStore.js
 *
 * Global client state for ZentraX AI, built with Zustand's "slices"
 * pattern: three independent slices (auth, chat, ui) combined into one
 * store so actions in one slice can safely read/reset another (e.g. logout
 * clearing chat state) without prop-drilling or context nesting.
 *
 * Persistence
 * -----------
 * Only non-sensitive, low-churn UI/profile state is persisted to
 * localStorage (`sidebarCollapsed`, `user`, `isAuthenticated`). Chat
 * messages and the auth token are deliberately excluded:
 *   - Messages are conversation content — persisting them to localStorage
 *     would work against a "privacy-first" posture and go stale the moment
 *     the server-side history diverges. They're refetched via
 *     `services/api.js` (`fetchChats` / `fetchChatHistory`) on load instead.
 *   - The auth token is already owned by `services/api.js`
 *     (`getAuthToken`/`setAuthToken`/`clearAuthToken`), which is the single
 *     source of truth Axios reads from on every request. Duplicating it
 *     into this store risks the two falling out of sync.
 *
 * Usage
 * -----
 *   import { useAppStore, useAuth, useChat, useUI } from "@/store/useAppStore";
 *
 *   const { user, login, logout } = useAuth();
 *   const { activeChatId, setActiveChat } = useChat();
 *   const { sidebarCollapsed, toggleSidebarCollapsed } = useUI();
 *
 * Prefer the `useAuth`/`useChat`/`useUI` selector hooks over pulling from
 * `useAppStore` directly — they scope re-renders to just that slice.
 */

import { create } from "zustand";
import { devtools, persist } from "zustand/middleware";
import { login as apiLogin, logout as apiLogout, setAuthToken, clearAuthToken } from "@/services/api";

// --------------------------------------------------------------------------- //
// Auth slice
// --------------------------------------------------------------------------- //

const createAuthSlice = (set, get) => ({
  user: null,
  isAuthenticated: false,
  isAuthLoading: false,
  authError: null,

  /**
   * Log in with credentials, persist the session token via services/api.js,
   * and store the returned user profile in state.
   * @param {{ email: string, password: string }} credentials
   */
  login: async (credentials) => {
    set({ isAuthLoading: true, authError: null }, false, "auth/login/start");
    try {
      const { user } = await apiLogin(credentials);
      set(
        { user, isAuthenticated: true, isAuthLoading: false },
        false,
        "auth/login/success"
      );
      return user;
    } catch (error) {
      set(
        { isAuthLoading: false, authError: error.message || "Login failed" },
        false,
        "auth/login/error"
      );
      throw error;
    }
  },

  /**
   * Log out both server-side and locally, and wipe in-memory chat state —
   * conversation content should not survive into the next signed-in session.
   */
  logout: async () => {
    try {
      await apiLogout();
    } catch {
      // Local state is cleared regardless of whether the server call
      // succeeds; the user should never be "stuck" logged in on the client.
      clearAuthToken();
    } finally {
      set(
        { user: null, isAuthenticated: false, authError: null },
        false,
        "auth/logout"
      );
      get().resetChatState();
    }
  },

  /** Directly set the user profile, e.g. after fetchCurrentUser() on app load. */
  setUser: (user) =>
    set({ user, isAuthenticated: Boolean(user) }, false, "auth/setUser"),

  /** Manually set a session token outside the normal login() flow (e.g. OAuth callback). */
  setSessionToken: (token) => {
    setAuthToken(token);
    set({ isAuthenticated: true }, false, "auth/setSessionToken");
  },

  clearAuthError: () => set({ authError: null }, false, "auth/clearError"),
});

// --------------------------------------------------------------------------- //
// Chat slice
// --------------------------------------------------------------------------- //

const createChatSlice = (set, get) => ({
  chats: [],
  activeChatId: null,
  messagesByChat: {},
  isSending: false,
  chatError: null,

  /** Replace the full chat list (e.g. after fetchChats()). */
  setChats: (chats) => set({ chats }, false, "chat/setChats"),

  /** Add a newly created chat and make it the active one. */
  addChat: (chat) =>
    set(
      (state) => ({ chats: [chat, ...state.chats], activeChatId: chat.id }),
      false,
      "chat/addChat"
    ),

  removeChat: (chatId) =>
    set(
      (state) => {
        const { [chatId]: _removed, ...remainingMessages } = state.messagesByChat;
        const remainingChats = state.chats.filter((c) => c.id !== chatId);
        const wasActive = state.activeChatId === chatId;
        return {
          chats: remainingChats,
          messagesByChat: remainingMessages,
          activeChatId: wasActive ? remainingChats[0]?.id ?? null : state.activeChatId,
        };
      },
      false,
      "chat/removeChat"
    ),

  setActiveChat: (chatId) => set({ activeChatId: chatId }, false, "chat/setActiveChat"),

  /** Replace the full message list for one chat (e.g. after fetchChatHistory()). */
  setMessages: (chatId, messages) =>
    set(
      (state) => ({ messagesByChat: { ...state.messagesByChat, [chatId]: messages } }),
      false,
      "chat/setMessages"
    ),

  /** Append a single message (user or assistant) to a chat's history. */
  addMessage: (chatId, message) =>
    set(
      (state) => ({
        messagesByChat: {
          ...state.messagesByChat,
          [chatId]: [...(state.messagesByChat[chatId] ?? []), message],
        },
      }),
      false,
      "chat/addMessage"
    ),

  setIsSending: (isSending) => set({ isSending }, false, "chat/setIsSending"),

  setChatError: (chatError) => set({ chatError }, false, "chat/setChatError"),

  /** Clear all chat/message state, e.g. on logout. Does not affect UI or auth slices. */
  resetChatState: () =>
    set(
      { chats: [], activeChatId: null, messagesByChat: {}, isSending: false, chatError: null },
      false,
      "chat/reset"
    ),
});

// --------------------------------------------------------------------------- //
// UI slice
// --------------------------------------------------------------------------- //

const createUISlice = (set) => ({
  sidebarCollapsed: false,
  isSettingsOpen: false,

  setSidebarCollapsed: (collapsed) =>
    set({ sidebarCollapsed: collapsed }, false, "ui/setSidebarCollapsed"),

  toggleSidebarCollapsed: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed }), false, "ui/toggleSidebar"),

  openSettings: () => set({ isSettingsOpen: true }, false, "ui/openSettings"),
  closeSettings: () => set({ isSettingsOpen: false }, false, "ui/closeSettings"),

  resetUIState: () =>
    set({ sidebarCollapsed: false, isSettingsOpen: false }, false, "ui/reset"),
});

// --------------------------------------------------------------------------- //
// Combined store
// --------------------------------------------------------------------------- //

export const useAppStore = create()(
  devtools(
    persist(
      (set, get) => ({
        ...createAuthSlice(set, get),
        ...createChatSlice(set, get),
        ...createUISlice(set, get),
      }),
      {
        name: "zentrax-app-store",
        // Only persist small, non-sensitive state — see the file header for why
        // messages and the auth token are intentionally left out.
        partialize: (state) => ({
          user: state.user,
          isAuthenticated: state.isAuthenticated,
          sidebarCollapsed: state.sidebarCollapsed,
        }),
      }
    ),
    { name: "ZentraXAppStore" }
  )
);

// --------------------------------------------------------------------------- //
// Scoped selector hooks
// --------------------------------------------------------------------------- //
// Components should prefer these over reading directly from useAppStore, so a
// re-render caused by a chat message doesn't also re-render something that
// only cares about sidebarCollapsed.

export const useAuth = () =>
  useAppStore((state) => ({
    user: state.user,
    isAuthenticated: state.isAuthenticated,
    isAuthLoading: state.isAuthLoading,
    authError: state.authError,
    login: state.login,
    logout: state.logout,
    setUser: state.setUser,
    setSessionToken: state.setSessionToken,
    clearAuthError: state.clearAuthError,
  }));

export const useChat = () =>
  useAppStore((state) => ({
    chats: state.chats,
    activeChatId: state.activeChatId,
    messagesByChat: state.messagesByChat,
    isSending: state.isSending,
    chatError: state.chatError,
    setChats: state.setChats,
    addChat: state.addChat,
    removeChat: state.removeChat,
    setActiveChat: state.setActiveChat,
    setMessages: state.setMessages,
    addMessage: state.addMessage,
    setIsSending: state.setIsSending,
    setChatError: state.setChatError,
    resetChatState: state.resetChatState,
  }));

export const useUI = () =>
  useAppStore((state) => ({
    sidebarCollapsed: state.sidebarCollapsed,
    isSettingsOpen: state.isSettingsOpen,
    setSidebarCollapsed: state.setSidebarCollapsed,
    toggleSidebarCollapsed: state.toggleSidebarCollapsed,
    openSettings: state.openSettings,
    closeSettings: state.closeSettings,
    resetUIState: state.resetUIState,
  }));