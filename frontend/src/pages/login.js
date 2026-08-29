/**
 * frontend/src/pages/login.js
 *
 * Authentication page for ZentraX AI (Next.js Pages Router).
 *
 * Design intent
 * -------------
 * Same token system as the rest of the app (#0B0E14 base, indigo accent,
 * teal "encrypted" trust signal), centered in a quiet full-viewport frame
 * with a faint radial vignette instead of a busy background — the card
 * itself, not the backdrop, should hold attention on an auth screen.
 *
 * Behavior
 * --------
 * - Client-side validation runs on submit and on blur/change once a field
 *   has been touched, with inline error text and a red-tinted input ring
 *   rather than a generic browser tooltip.
 * - Password visibility toggle.
 * - Submit button shows a spinner and disables while the (placeholder)
 *   auth call is in flight; a top-level server error banner is reserved
 *   for auth failures distinct from field-level validation errors.
 * - Links to the signup page and a forgot-password flow.
 */

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { Eye, EyeOff, Loader2, ShieldCheck } from "lucide-react";

const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

/**
 * Integration point: replace with a real call to your FastAPI auth
 * endpoint, e.g. POST /api/auth/login with { email, password }.
 */
async function loginUser({ email, password }) {
  await new Promise((resolve) => setTimeout(resolve, 900));
  if (password.length < 8) {
    const error = new Error("Incorrect email or password.");
    error.code = "INVALID_CREDENTIALS";
    throw error;
  }
  return { token: "placeholder-session-token" };
}

function FieldError({ message }) {
  if (!message) return null;
  return (
    <p id="field-error" className="mt-1.5 text-xs text-red-400" role="alert">
      {message}
    </p>
  );
}

export default function LoginPage() {
  const router = useRouter();

  const [values, setValues] = useState({ email: "", password: "" });
  const [touched, setTouched] = useState({});
  const [showPassword, setShowPassword] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [serverError, setServerError] = useState("");

  const validate = (fieldValues = values) => {
    const errors = {};
    if (!fieldValues.email.trim()) {
      errors.email = "Email is required.";
    } else if (!EMAIL_PATTERN.test(fieldValues.email.trim())) {
      errors.email = "Enter a valid email address.";
    }

    if (!fieldValues.password) {
      errors.password = "Password is required.";
    } else if (fieldValues.password.length < 8) {
      errors.password = "Password must be at least 8 characters.";
    }
    return errors;
  };

  const errors = validate();

  const handleChange = (field) => (e) => {
    setValues((prev) => ({ ...prev, [field]: e.target.value }));
    if (serverError) setServerError("");
  };

  const handleBlur = (field) => () => {
    setTouched((prev) => ({ ...prev, [field]: true }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTouched({ email: true, password: true });

    const currentErrors = validate();
    if (Object.keys(currentErrors).length > 0) return;

    setIsSubmitting(true);
    setServerError("");
    try {
      await loginUser(values);
      router.push("/");
    } catch (error) {
      setServerError(error.message || "Something went wrong. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const showEmailError = touched.email && errors.email;
  const showPasswordError = touched.password && errors.password;

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#0B0E14] px-4">
      {/* Ambient vignette, not a busy background */}
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top,_rgba(99,102,241,0.10),_transparent_60%)]"
        aria-hidden="true"
      />

      <div className="relative w-full max-w-sm">
        {/* Brand mark */}
        <div className="mb-8 flex flex-col items-center gap-3">
          <div className="relative flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500/20 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/20">
            <span className="font-display text-base font-bold text-indigo-300">Z</span>
            <span
              className="motion-safe:animate-pulse absolute -right-0.5 -top-0.5 h-2.5 w-2.5 rounded-full bg-teal-400 ring-2 ring-[#0B0E14]"
              aria-hidden="true"
            />
          </div>
          <p className="font-display text-base font-semibold tracking-tight text-slate-100">
            ZentraX <span className="text-slate-500">AI</span>
          </p>
        </div>

        {/* Auth card */}
        <div className="rounded-2xl border border-white/10 bg-white/[0.02] p-6 shadow-2xl shadow-black/40 backdrop-blur-sm sm:p-7">
          <h1 className="text-lg font-semibold text-slate-100">Sign in</h1>
          <p className="mt-1 text-sm text-slate-500">Welcome back to your encrypted workspace.</p>

          {serverError && (
            <div
              role="alert"
              className="mt-4 rounded-lg border border-red-400/20 bg-red-500/10 px-3 py-2 text-sm text-red-300"
            >
              {serverError}
            </div>
          )}

          <form onSubmit={handleSubmit} noValidate className="mt-6 space-y-4">
            <div>
              <label htmlFor="email" className="mb-1.5 block text-xs font-medium text-slate-400">
                Email
              </label>
              <input
                id="email"
                type="email"
                autoComplete="email"
                value={values.email}
                onChange={handleChange("email")}
                onBlur={handleBlur("email")}
                aria-invalid={Boolean(showEmailError)}
                aria-describedby={showEmailError ? "field-error" : undefined}
                placeholder="you@company.com"
                className={[
                  "w-full rounded-lg border bg-white/[0.03] px-3 py-2.5 text-sm text-slate-100",
                  "placeholder:text-slate-600 focus:outline-none focus:ring-2",
                  "transition-colors duration-150",
                  showEmailError
                    ? "border-red-400/40 focus:ring-red-400/40"
                    : "border-white/10 focus:border-indigo-400/40 focus:ring-indigo-400/40",
                ].join(" ")}
              />
              <FieldError message={showEmailError ? errors.email : ""} />
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <label htmlFor="password" className="block text-xs font-medium text-slate-400">
                  Password
                </label>
                <Link
                  href="/forgot-password"
                  className="text-xs text-slate-500 transition-colors duration-150 hover:text-indigo-300"
                >
                  Forgot password?
                </Link>
              </div>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  value={values.password}
                  onChange={handleChange("password")}
                  onBlur={handleBlur("password")}
                  aria-invalid={Boolean(showPasswordError)}
                  aria-describedby={showPasswordError ? "field-error" : undefined}
                  placeholder="••••••••"
                  className={[
                    "w-full rounded-lg border bg-white/[0.03] px-3 py-2.5 pr-10 text-sm text-slate-100",
                    "placeholder:text-slate-600 focus:outline-none focus:ring-2",
                    "transition-colors duration-150",
                    showPasswordError
                      ? "border-red-400/40 focus:ring-red-400/40"
                      : "border-white/10 focus:border-indigo-400/40 focus:ring-indigo-400/40",
                  ].join(" ")}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((v) => !v)}
                  tabIndex={-1}
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 transition-colors duration-150 hover:text-slate-300"
                >
                  {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
              <FieldError message={showPasswordError ? errors.password : ""} />
            </div>

            <button
              type="submit"
              disabled={isSubmitting}
              className={[
                "flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5 text-sm font-medium",
                "transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
                isSubmitting
                  ? "cursor-not-allowed bg-indigo-500/50 text-white/70"
                  : "bg-indigo-500 text-white hover:bg-indigo-400",
              ].join(" ")}
            >
              {isSubmitting && <Loader2 className="h-4 w-4 animate-spin" />}
              {isSubmitting ? "Signing in…" : "Sign in"}
            </button>
          </form>

          <div className="mt-4 flex items-center justify-center gap-1.5 text-[11px] text-slate-600">
            <ShieldCheck className="h-3.5 w-3.5 text-teal-400/80" />
            Protected by end-to-end encryption
          </div>
        </div>

        <p className="mt-6 text-center text-sm text-slate-500">
          Don&apos;t have an account?{" "}
          <Link href="/signup" className="font-medium text-indigo-300 hover:text-indigo-200">
            Create one
          </Link>
        </p>
      </div>
    </div>
  );
}