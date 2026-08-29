"use client";

/**
 * frontend/src/components/Navbar.jsx
 *
 * Top navigation bar for ZentraX AI.
 *
 * Design intent
 * -------------
 * Same token system as Sidebar.jsx / ChatBox.jsx: deep charcoal-navy
 * (#0B0E14), a single indigo accent for the active link and primary
 * actions, and the muted teal "encrypted" signal carried over from the
 * brand mark so the trust cue stays consistent everywhere it appears.
 * A hairline bottom border (not a shadow) separates the bar from content,
 * keeping the surface flat and quiet.
 *
 * Behavior
 * --------
 * - Desktop (md and up): logo left, inline nav links, encryption badge and
 *   profile menu right.
 * - Mobile: logo left, profile avatar + menu button right; nav links move
 *   into a slide-down panel toggled by the menu button.
 * - Active link is either passed explicitly per-item (`active: true`) or,
 *   if omitted, derived from the current path via `usePathname`.
 * - Profile menu is a simple controlled dropdown, closes on outside click,
 *   Escape, or selecting an action.
 */

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ChevronDown, LogOut, Menu, Settings, ShieldCheck, User, X } from "lucide-react";

function BrandMark({ encryptionActive }) {
  return (
    <Link
      href="/"
      className="flex items-center gap-2.5 rounded-md focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60"
    >
      <div className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/20">
        <span className="font-display text-sm font-bold text-indigo-300">Z</span>
        {encryptionActive && (
          <span
            className="motion-safe:animate-pulse absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-teal-400 ring-2 ring-[#0B0E14]"
            aria-hidden="true"
          />
        )}
      </div>
      <span className="font-display text-sm font-semibold tracking-tight text-slate-100">
        ZentraX <span className="text-slate-500">AI</span>
      </span>
    </Link>
  );
}

function NavLink({ href, label, active, onClick }) {
  return (
    <Link
      href={href}
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      className={[
        "rounded-md px-3 py-1.5 text-sm transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
        active
          ? "bg-indigo-500/10 text-indigo-200"
          : "text-slate-400 hover:bg-white/5 hover:text-slate-100",
      ].join(" ")}
    >
      {label}
    </Link>
  );
}

function ProfileMenu({ user, onOpenSettings, onLogout }) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const handleClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) setOpen(false);
    };
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", handleClick);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("mousedown", handleClick);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [open]);

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded-full p-1 pr-2 text-slate-300 transition-colors duration-150 hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60"
      >
        <div className="flex h-7 w-7 items-center justify-center rounded-full bg-white/10 text-xs font-semibold text-slate-200">
          {user?.initials || user?.name?.[0] || <User className="h-3.5 w-3.5" />}
        </div>
        <ChevronDown className={["h-3.5 w-3.5 transition-transform duration-150", open ? "rotate-180" : ""].join(" ")} />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 top-full z-50 mt-2 w-48 overflow-hidden rounded-xl border border-white/10 bg-[#12161F] py-1 shadow-xl"
        >
          {user && (
            <div className="border-b border-white/[0.06] px-3 py-2">
              <p className="truncate text-sm text-slate-200">{user.name}</p>
              {user.email && <p className="truncate text-xs text-slate-500">{user.email}</p>}
            </div>
          )}
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              onOpenSettings();
              setOpen(false);
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-slate-300 transition-colors duration-150 hover:bg-white/5 hover:text-slate-100"
          >
            <Settings className="h-4 w-4 text-slate-500" />
            Settings
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => {
              onLogout();
              setOpen(false);
            }}
            className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm text-slate-300 transition-colors duration-150 hover:bg-white/5 hover:text-red-300"
          >
            <LogOut className="h-4 w-4 text-slate-500" />
            Log out
          </button>
        </div>
      )}
    </div>
  );
}

export default function Navbar({
  links = [],
  activeHref,
  encryptionActive = true,
  user = null,
  onOpenSettings = () => {},
  onLogout = () => {},
  className = "",
}) {
  const pathname = usePathname();
  const resolvedActiveHref = activeHref ?? pathname;

  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className={["sticky top-0 z-40 border-b border-white/[0.06] bg-[#0B0E14]", className].join(" ")}>
      <div className="flex h-14 items-center justify-between px-4">
        <BrandMark encryptionActive={encryptionActive} />

        {/* Desktop nav links */}
        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {links.map((link) => (
            <NavLink
              key={link.href}
              href={link.href}
              label={link.label}
              active={link.active ?? link.href === resolvedActiveHref}
            />
          ))}
        </nav>

        {/* Right side */}
        <div className="flex items-center gap-3">
          {encryptionActive && (
            <div className="hidden items-center gap-1 text-[11px] text-teal-400/90 sm:flex">
              <ShieldCheck className="h-3.5 w-3.5" />
              <span>Encrypted</span>
            </div>
          )}

          <div className="hidden md:block">
            <ProfileMenu user={user} onOpenSettings={onOpenSettings} onLogout={onLogout} />
          </div>

          {/* Mobile menu toggle */}
          <button
            type="button"
            onClick={() => setMobileOpen((v) => !v)}
            aria-expanded={mobileOpen}
            aria-label={mobileOpen ? "Close menu" : "Open menu"}
            className="rounded-md p-1.5 text-slate-300 transition-colors duration-150 hover:bg-white/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60 md:hidden"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile panel */}
      <div
        className={[
          "grid overflow-hidden border-t border-white/[0.06] transition-[grid-template-rows] duration-200 ease-out md:hidden",
          mobileOpen ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
        ].join(" ")}
      >
        <div className="min-h-0">
          <nav aria-label="Primary" className="flex flex-col gap-0.5 px-3 py-2">
            {links.map((link) => (
              <NavLink
                key={link.href}
                href={link.href}
                label={link.label}
                active={link.active ?? link.href === resolvedActiveHref}
                onClick={() => setMobileOpen(false)}
              />
            ))}
          </nav>

          <div className="border-t border-white/[0.06] px-3 py-2">
            {user && (
              <div className="flex items-center gap-2.5 px-1 py-2">
                <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs font-semibold text-slate-200">
                  {user.initials || user.name?.[0] || "U"}
                </div>
                <div className="min-w-0">
                  <p className="truncate text-sm text-slate-200">{user.name}</p>
                  {user.email && <p className="truncate text-xs text-slate-500">{user.email}</p>}
                </div>
              </div>
            )}
            <button
              type="button"
              onClick={() => {
                onOpenSettings();
                setMobileOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm text-slate-300 hover:bg-white/5 hover:text-slate-100"
            >
              <Settings className="h-4 w-4 text-slate-500" />
              Settings
            </button>
            <button
              type="button"
              onClick={() => {
                onLogout();
                setMobileOpen(false);
              }}
              className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm text-slate-300 hover:bg-white/5 hover:text-red-300"
            >
              <LogOut className="h-4 w-4 text-slate-500" />
              Log out
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
