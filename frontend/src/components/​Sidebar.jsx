"use client";

/**
 * frontend/src/components/Sidebar.jsx
 *
 * Primary navigation sidebar for ZentraX AI.
 *
 * Design intent
 * -------------
 * Deep charcoal-navy surface (never pure black) with a single indigo accent
 * reserved for active/selected states, and a muted teal reserved only for
 * the "encryption active" signal — the one piece of brand truth this
 * product needs to keep visible at all times. Section labels use a small
 * uppercase utility style to read as system chrome, not marketing copy.
 *
 * Behavior
 * --------
 * - Desktop (lg and up): persistent rail that toggles between an expanded
 *   (264px) and collapsed (72px, icon-only) state.
 * - Mobile / tablet: hidden off-canvas by default, opened as an overlay
 *   drawer via the built-in menu trigger, closed on backdrop click, Escape,
 *   or selecting a chat/tool.
 * - Collapse state can be uncontrolled (internal) or controlled via
 *   `collapsed` + `onCollapsedChange`.
 *
 * Suggested Tailwind theme additions (optional, component works without them):
 *   fontFamily: {
 *     display: ['"Space Grotesk"', 'sans-serif'],
 *     sans: ['Inter', 'sans-serif'],
 *   }
 */

import { useEffect, useRef, useState } from "react";
import {
  MessageSquare,
  Plus,
  Wrench,
  Settings,
  ChevronsLeft,
  ChevronsRight,
  Menu,
  X,
  ShieldCheck,
} from "lucide-react";

function SectionLabel({ children, collapsed }) {
  if (collapsed) return null;
  return (
    <div className="px-3 pt-5 pb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
      {children}
    </div>
  );
}

function RailButton({ icon: Icon, label, collapsed, onClick, ...props }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={collapsed ? label : undefined}
      className={[
        "group flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-300",
        "transition-colors duration-150",
        "hover:bg-white/5 hover:text-slate-100",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
        collapsed ? "justify-center" : "",
      ].join(" ")}
      {...props}
    >
      <Icon className="h-[18px] w-[18px] shrink-0 text-slate-400 group-hover:text-slate-100" />
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  );
}

function NavItem({ icon: Icon, label, active, collapsed, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      title={collapsed ? label : undefined}
      className={[
        "group relative flex w-full items-center gap-3 rounded-lg px-3 py-2 text-sm",
        "transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
        collapsed ? "justify-center" : "",
        active
          ? "bg-indigo-500/10 text-indigo-200"
          : "text-slate-300 hover:bg-white/5 hover:text-slate-100",
      ].join(" ")}
    >
      {active && (
        <span className="absolute left-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-r-full bg-indigo-400" />
      )}
      <Icon
        className={[
          "h-[18px] w-[18px] shrink-0",
          active ? "text-indigo-300" : "text-slate-400 group-hover:text-slate-100",
        ].join(" ")}
      />
      {!collapsed && <span className="truncate">{label}</span>}
    </button>
  );
}

function ChatListItem({ chat, active, collapsed, onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-current={active ? "page" : undefined}
      title={collapsed ? chat.title : undefined}
      className={[
        "group flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-left text-sm",
        "transition-colors duration-150",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
        collapsed ? "justify-center" : "",
        active
          ? "bg-white/[0.06] text-slate-100"
          : "text-slate-400 hover:bg-white/5 hover:text-slate-200",
      ].join(" ")}
    >
      <MessageSquare className="h-4 w-4 shrink-0 text-slate-500 group-hover:text-slate-300" />
      {!collapsed && <span className="truncate">{chat.title}</span>}
    </button>
  );
}

export default function Sidebar({
  chats = [],
  activeChatId = null,
  onSelectChat = () => {},
  onNewChat = () => {},

  tools = [],
  activeToolId = null,
  onSelectTool = () => {},

  onOpenSettings = () => {},

  user = null,
  encryptionActive = true,

  collapsed: collapsedProp,
  defaultCollapsed = false,
  onCollapsedChange,

  className = "",
}) {
  const [internalCollapsed, setInternalCollapsed] = useState(defaultCollapsed);
  const collapsed = collapsedProp !== undefined ? collapsedProp : internalCollapsed;

  const setCollapsed = (next) => {
    if (onCollapsedChange) onCollapsedChange(next);
    if (collapsedProp === undefined) setInternalCollapsed(next);
  };

  const [mobileOpen, setMobileOpen] = useState(false);
  const closeButtonRef = useRef(null);

  // Close the mobile drawer on Escape.
  useEffect(() => {
    if (!mobileOpen) return;
    const handleKeyDown = (e) => {
      if (e.key === "Escape") setMobileOpen(false);
    };
    document.addEventListener("keydown", handleKeyDown);
    closeButtonRef.current?.focus();
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [mobileOpen]);

  const handleSelectChat = (id) => {
    onSelectChat(id);
    setMobileOpen(false);
  };

  const handleSelectTool = (id) => {
    onSelectTool(id);
    setMobileOpen(false);
  };

  const handleOpenSettings = () => {
    onOpenSettings();
    setMobileOpen(false);
  };

  const sidebarBody = (
    <div className="flex h-full flex-col bg-[#0B0E14]">
      {/* Brand header */}
      <div
        className={[
          "flex items-center gap-2.5 border-b border-white/[0.06] px-3 py-4",
          collapsed ? "justify-center" : "justify-between",
        ].join(" ")}
      >
        <div className={["flex items-center gap-2.5", collapsed ? "" : "min-w-0"].join(" ")}>
          <div className="relative flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/20 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/20">
            <span className="font-display text-sm font-bold text-indigo-300">Z</span>
            {encryptionActive && (
              <span
                className="motion-safe:animate-pulse absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-teal-400 ring-2 ring-[#0B0E14]"
                aria-hidden="true"
              />
            )}
          </div>
          {!collapsed && (
            <div className="min-w-0">
              <p className="truncate font-display text-sm font-semibold tracking-tight text-slate-100">
                ZentraX AI
              </p>
              {encryptionActive && (
                <p className="flex items-center gap-1 text-[11px] text-teal-400/90">
                  <ShieldCheck className="h-3 w-3" />
                  Encrypted session
                </p>
              )}
            </div>
          )}
        </div>

        {/* Mobile close */}
        <button
          ref={closeButtonRef}
          type="button"
          onClick={() => setMobileOpen(false)}
          className="rounded-md p-1.5 text-slate-400 hover:bg-white/5 hover:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60 lg:hidden"
          aria-label="Close sidebar"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* New chat */}
      <div className="px-2 pt-3">
        <button
          type="button"
          onClick={() => {
            onNewChat();
            setMobileOpen(false);
          }}
          title={collapsed ? "New chat" : undefined}
          className={[
            "flex w-full items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2",
            "text-sm font-medium text-slate-200 transition-colors duration-150",
            "hover:border-indigo-400/40 hover:bg-indigo-500/10 hover:text-indigo-200",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
            collapsed ? "justify-center" : "",
          ].join(" ")}
        >
          <Plus className="h-4 w-4 shrink-0" />
          {!collapsed && <span>New chat</span>}
        </button>
      </div>

      {/* Scrollable middle: chats + tools */}
      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        <SectionLabel collapsed={collapsed}>Chats</SectionLabel>
        <div className="space-y-0.5">
          {chats.length === 0 && !collapsed && (
            <p className="px-3 py-2 text-xs text-slate-500">No chats yet — start one above.</p>
          )}
          {chats.map((chat) => (
            <ChatListItem
              key={chat.id}
              chat={chat}
              collapsed={collapsed}
              active={chat.id === activeChatId}
              onClick={() => handleSelectChat(chat.id)}
            />
          ))}
        </div>

        {tools.length > 0 && (
          <>
            <SectionLabel collapsed={collapsed}>Tools</SectionLabel>
            <div className="space-y-0.5">
              {tools.map((tool) => (
                <NavItem
                  key={tool.id}
                  icon={tool.icon || Wrench}
                  label={tool.label}
                  collapsed={collapsed}
                  active={tool.id === activeToolId}
                  onClick={() => handleSelectTool(tool.id)}
                />
              ))}
            </div>
          </>
        )}
      </div>

      {/* Footer: settings, user, collapse toggle */}
      <div className="border-t border-white/[0.06] px-2 py-2">
        <RailButton icon={Settings} label="Settings" collapsed={collapsed} onClick={handleOpenSettings} />

        {user && (
          <div
            className={[
              "mt-1 flex items-center gap-2.5 rounded-lg px-3 py-2",
              collapsed ? "justify-center" : "",
            ].join(" ")}
          >
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs font-semibold text-slate-200">
              {user.initials || user.name?.[0] || "U"}
            </div>
            {!collapsed && (
              <div className="min-w-0">
                <p className="truncate text-sm text-slate-200">{user.name}</p>
                {user.email && (
                  <p className="truncate text-xs text-slate-500">{user.email}</p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Desktop collapse toggle */}
        <button
          type="button"
          onClick={() => setCollapsed(!collapsed)}
          className={[
            "mt-1 hidden w-full items-center gap-3 rounded-lg px-3 py-2 text-sm text-slate-400 lg:flex",
            "transition-colors duration-150 hover:bg-white/5 hover:text-slate-100",
            "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
            collapsed ? "justify-center" : "",
          ].join(" ")}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronsRight className="h-4 w-4" /> : <ChevronsLeft className="h-4 w-4" />}
          {!collapsed && <span>Collapse</span>}
        </button>
      </div>
    </div>
  );

  return (
    <>
      {/* Mobile trigger (rendered only when drawer is closed) */}
      {!mobileOpen && (
        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          className="fixed left-3 top-3 z-40 rounded-md border border-white/10 bg-[#0B0E14] p-2 text-slate-300 shadow-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60 lg:hidden"
          aria-label="Open sidebar"
        >
          <Menu className="h-5 w-5" />
        </button>
      )}

      {/* Mobile backdrop */}
      {mobileOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
          onClick={() => setMobileOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Mobile drawer */}
      <nav
        aria-label="Primary"
        className={[
          "fixed inset-y-0 left-0 z-50 w-[264px] transform transition-transform duration-200 ease-out lg:hidden",
          mobileOpen ? "translate-x-0" : "-translate-x-full",
        ].join(" ")}
      >
        {sidebarBody}
      </nav>

      {/* Desktop rail */}
      <nav
        aria-label="Primary"
        className={[
          "sticky top-0 hidden h-screen shrink-0 transition-[width] duration-200 ease-out lg:block",
          collapsed ? "w-[72px]" : "w-[264px]",
          className,
        ].join(" ")}
      >
        {sidebarBody}
      </nav>
    </>
  );
}