"use client";

/**
 * frontend/src/components/ChatBox.jsx
 *
 * Primary chat surface for ZentraX AI.
 *
 * Design intent
 * -------------
 * Shares the same token system as Sidebar.jsx: deep charcoal-navy surface
 * (#0B0E14), a single indigo accent for the user's own messages and active
 * affordances, and a muted teal reserved for the "encrypted" trust signal.
 * Assistant messages sit on a barely-lifted panel tone rather than a bubble
 * outline, so the transcript reads as a continuous, calm surface rather
 * than a stack of chat-app cartoon bubbles.
 *
 * Behavior
 * --------
 * - Renders a scrollable message history (user vs. assistant styling) and
 *   auto-scrolls to the latest message as new ones arrive.
 * - Bottom input is a growing textarea: Enter sends, Shift+Enter inserts a
 *   newline, and the send button / Enter are disabled while `isLoading` or
 *   the input is empty.
 * - A three-dot typing indicator renders as its own row while a response
 *   is pending.
 * - Message list and input value can be fully controlled by the parent, or
 *   the input value can be left uncontrolled and managed internally.
 */

import { useEffect, useRef, useState } from "react";
import { Send, ShieldCheck, Sparkles, User } from "lucide-react";

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 px-1 py-2" aria-label="Assistant is typing">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="h-1.5 w-1.5 rounded-full bg-slate-500 motion-safe:animate-bounce"
          style={{ animationDelay: `${i * 120}ms` }}
        />
      ))}
    </div>
  );
}

function Avatar({ role }) {
  if (role === "user") {
    return (
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/10 text-slate-200">
        <User className="h-3.5 w-3.5" />
      </div>
    );
  }
  return (
    <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500/25 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/25 text-indigo-300">
      <Sparkles className="h-3.5 w-3.5" />
    </div>
  );
}

function MessageBubble({ message }) {
  const isUser = message.role === "user";
  return (
    <div className={["flex gap-3", isUser ? "flex-row-reverse" : "flex-row"].join(" ")}>
      <Avatar role={message.role} />
      <div className={["flex max-w-[75%] flex-col gap-1", isUser ? "items-end" : "items-start"].join(" ")}>
        <div
          className={[
            "whitespace-pre-wrap break-words rounded-2xl px-4 py-2.5 text-sm leading-relaxed",
            isUser
              ? "rounded-tr-sm bg-indigo-500/15 text-slate-100 ring-1 ring-inset ring-indigo-400/20"
              : "rounded-tl-sm bg-white/[0.04] text-slate-200 ring-1 ring-inset ring-white/[0.06]",
          ].join(" ")}
        >
          {message.content}
        </div>
        {message.timestamp && (
          <span className="px-1 text-[11px] text-slate-500">{message.timestamp}</span>
        )}
      </div>
    </div>
  );
}

function EmptyState({ assistantName }) {
  return (
    <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-500/25 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/25">
        <Sparkles className="h-5 w-5 text-indigo-300" />
      </div>
      <p className="text-sm font-medium text-slate-200">Start a conversation with {assistantName}</p>
      <p className="max-w-xs text-xs text-slate-500">
        Messages in this chat are processed under your workspace's encryption settings.
      </p>
    </div>
  );
}

export default function ChatBox({
  messages = [],
  onSendMessage = () => {},
  isLoading = false,
  assistantName = "ZentraX",
  placeholder = "Message ZentraX…",
  encryptionActive = true,
  inputValue: inputValueProp,
  onInputChange,
  disabled = false,
  className = "",
}) {
  const [internalInput, setInternalInput] = useState("");
  const inputValue = inputValueProp !== undefined ? inputValueProp : internalInput;

  const setInputValue = (next) => {
    if (onInputChange) onInputChange(next);
    if (inputValueProp === undefined) setInternalInput(next);
  };

  const scrollRef = useRef(null);
  const textareaRef = useRef(null);

  // Auto-scroll to the latest message / typing indicator.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, isLoading]);

  // Auto-grow the textarea up to a max height.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [inputValue]);

  const canSend = inputValue.trim().length > 0 && !isLoading && !disabled;

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!canSend) return;
    onSendMessage(inputValue.trim());
    setInputValue("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className={["flex h-full min-h-0 flex-col bg-[#0B0E14]", className].join(" ")}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500/25 to-indigo-500/5 ring-1 ring-inset ring-indigo-400/25">
            <Sparkles className="h-3.5 w-3.5 text-indigo-300" />
          </div>
          <p className="text-sm font-medium text-slate-100">{assistantName}</p>
        </div>
        {encryptionActive && (
          <div className="flex items-center gap-1 text-[11px] text-teal-400/90">
            <ShieldCheck className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Encrypted</span>
          </div>
        )}
      </div>

      {/* Message history */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 ? (
          <EmptyState assistantName={assistantName} />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {isLoading && (
              <div className="flex gap-3">
                <Avatar role="assistant" />
                <TypingIndicator />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Input */}
      <div className="border-t border-white/[0.06] px-4 py-3">
        <form onSubmit={handleSubmit} className="mx-auto flex max-w-3xl items-end gap-2">
          <div className="flex min-w-0 flex-1 items-end rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-2 transition-colors duration-150 focus-within:border-indigo-400/40">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              disabled={disabled}
              rows={1}
              className={[
                "max-h-40 w-full resize-none bg-transparent text-sm text-slate-100",
                "placeholder:text-slate-500 focus:outline-none",
                "disabled:cursor-not-allowed disabled:opacity-50",
              ].join(" ")}
            />
          </div>
          <button
            type="submit"
            disabled={!canSend}
            aria-label="Send message"
            className={[
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-xl transition-colors duration-150",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400/60",
              canSend
                ? "bg-indigo-500 text-white hover:bg-indigo-400"
                : "cursor-not-allowed bg-white/5 text-slate-600",
            ].join(" ")}
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl px-1 text-[11px] text-slate-600">
          Press Enter to send, Shift + Enter for a new line.
        </p>
      </div>
    </div>
  );
}