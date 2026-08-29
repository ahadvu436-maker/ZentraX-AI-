"use client";

/**
 * frontend/src/app/page.js
 *
 * Main chat workspace for ZentraX AI — composes Navbar, Sidebar, and
 * ChatBox into a single full-viewport layout.
 *
 * Assumes the Next.js App Router (Navbar.jsx uses `usePathname` from
 * `next/navigation`, which is App Router-only). If this project uses the
 * Pages Router instead, swap that hook out for `useRouter` from
 * `next/router` inside Navbar.jsx and drop this file in as
 * `frontend/src/pages/index.js` unchanged otherwise.
 *
 * Layout
 * ------
 *   ┌───────────┬───────────────────────────────┐
 *   │           │            Navbar              │
 *   │  Sidebar  ├───────────────────────────────┤
 *   │  (rail)   │                                │
 *   │           │            ChatBox             │
 *   │           │                                │
 *   └───────────┴───────────────────────────────┘
 * Sidebar is a persistent rail on desktop (collapsible) and an off-canvas
 * drawer on mobile — that open/close state is self-contained inside
 * Sidebar.jsx. This page owns the *desktop* collapse state so it can react
 * to viewport size, which is the "layout responsiveness" concern that
 * actually needs to live above the component.
 */

import { useEffect, useState } from "react";
import { Settings, Wrench } from "lucide-react";
import Navbar from "@/components/Navbar";
import Sidebar from "@/components/Sidebar";
import ChatBox from "@/components/ChatBox";

const NAV_LINKS = [
  { href: "/", label: "Chat" },
  { href: "/tools", label: "Tools" },
  { href: "/docs", label: "Docs" },
];

const INITIAL_CHATS = [
  { id: "chat-1", title: "Onboarding walkthrough" },
  { id: "chat-2", title: "API integration questions" },
  { id: "chat-3", title: "Privacy policy draft" },
];

const INITIAL_MESSAGES = {
  "chat-1": [
    {
      id: "m1",
      role: "assistant",
      content: "Welcome to ZentraX AI. Ask me anything — this session is end-to-end encrypted.",
      timestamp: "9:00 AM",
    },
  ],
  "chat-2": [],
  "chat-3": [],
};

const TOOLS = [
  { id: "tool-search", label: "Web search", icon: Wrench },
  { id: "tool-files", label: "File analysis", icon: Wrench },
];

const CURRENT_USER = {
  name: "Jordan Alvarez",
  email: "jordan@zentrax.ai",
  initials: "JA",
};

// Desktop sidebar auto-collapses below this viewport width (px).
const COLLAPSE_BREAKPOINT = 1280;

/**
 * Integration point: replace this with a real call to your FastAPI
 * gateway, e.g. POST /api/chat with { chatId, message }, and stream or
 * await the assistant's reply.
 */
async function sendMessageToApi(chatId, message) {
  await new Promise((resolve) => setTimeout(resolve, 900));
  return {
    id: `${chatId}-${Date.now()}`,
    role: "assistant",
    content: `Got it — here's a placeholder reply to: "${message}"`,
    timestamp: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
  };
}

export default function HomePage() {
  const [chats] = useState(INITIAL_CHATS);
  const [activeChatId, setActiveChatId] = useState(INITIAL_CHATS[0].id);
  const [messagesByChat, setMessagesByChat] = useState(INITIAL_MESSAGES);
  const [isLoading, setIsLoading] = useState(false);
  const [activeToolId, setActiveToolId] = useState(null);

  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Auto-collapse the desktop sidebar rail on narrower viewports; the user
  // can still manually expand it afterward via Sidebar's own toggle.
  useEffect(() => {
    const mql = window.matchMedia(`(max-width: ${COLLAPSE_BREAKPOINT}px)`);
    const applyBreakpoint = (e) => setSidebarCollapsed(e.matches);

    applyBreakpoint(mql);
    mql.addEventListener("change", applyBreakpoint);
    return () => mql.removeEventListener("change", applyBreakpoint);
  }, []);

  const activeMessages = messagesByChat[activeChatId] ?? [];

  const handleNewChat = () => {
    // Integration point: create a new chat via your backend, then switch
    // `activeChatId` to the newly created id once it's returned.
    console.log("New chat requested");
  };

  const handleSelectChat = (chatId) => {
    setActiveChatId(chatId);
  };

  const handleSelectTool = (toolId) => {
    setActiveToolId((current) => (current === toolId ? null : toolId));
  };

  const handleSendMessage = async (text) => {
    const userMessage = {
      id: `${activeChatId}-${Date.now()}-user`,
      role: "user",
      content: text,
      timestamp: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
    };

    setMessagesByChat((prev) => ({
      ...prev,
      [activeChatId]: [...(prev[activeChatId] ?? []), userMessage],
    }));
    setIsLoading(true);

    try {
      const assistantMessage = await sendMessageToApi(activeChatId, text);
      setMessagesByChat((prev) => ({
        ...prev,
        [activeChatId]: [...(prev[activeChatId] ?? []), assistantMessage],
      }));
    } catch (error) {
      console.error("Failed to get a response", error);
      setMessagesByChat((prev) => ({
        ...prev,
        [activeChatId]: [
          ...(prev[activeChatId] ?? []),
          {
            id: `${activeChatId}-${Date.now()}-error`,
            role: "assistant",
            content: "Something went wrong reaching ZentraX. Please try again.",
          },
        ],
      }));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen min-h-0 w-full overflow-hidden bg-[#0B0E14]">
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        tools={TOOLS}
        activeToolId={activeToolId}
        onSelectTool={handleSelectTool}
        onOpenSettings={() => console.log("Open settings")}
        user={CURRENT_USER}
        encryptionActive
        collapsed={sidebarCollapsed}
        onCollapsedChange={setSidebarCollapsed}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar
          links={NAV_LINKS}
          encryptionActive
          user={CURRENT_USER}
          onOpenSettings={() => console.log("Open settings")}
          onLogout={() => console.log("Log out")}
        />

        <main className="min-h-0 flex-1">
          <ChatBox
            messages={activeMessages}
            onSendMessage={handleSendMessage}
            isLoading={isLoading}
            encryptionActive
          />
        </main>
      </div>
    </div>
  );
}