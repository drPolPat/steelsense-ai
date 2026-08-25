import { useEffect, useRef, useState } from 'react';
import { postChat } from '../api';
import type { ChatMessage } from '../types';

const SUGGESTIONS = [
  'Is Beam 3A showing signs of fatigue consistent with cyclic loading?',
  'Summarize the structural health of the bridge this week.',
  'Which sensor location shows the most concerning trend?',
];

export default function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [messages, pending]);

  async function send(question: string) {
    const trimmed = question.trim();
    if (!trimmed || pending) return;

    setMessages((prev) => [...prev, { role: 'user', content: trimmed }]);
    setInput('');
    setPending(true);
    setError(null);

    try {
      const { answer } = await postChat(trimmed);
      setMessages((prev) => [...prev, { role: 'assistant', content: answer }]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Something went wrong.');
    } finally {
      setPending(false);
    }
  }

  return (
    <div className="card chat-panel">
      <div className="card-header">Ask SteelSense AI</div>
      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Ask a question about the bridge's structural condition.</p>
            <div className="chat-suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} className="chat-suggestion" onClick={() => send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message, i) => (
          <div key={i} className={`chat-bubble ${message.role}`}>
            {message.content}
          </div>
        ))}
        {pending && <div className="chat-bubble assistant pending">Thinking…</div>}
        {error && <div className="chat-bubble error">{error}</div>}
      </div>
      <form
        className="chat-input-row"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about a sensor, trend, or the bridge overall…"
          disabled={pending}
        />
        <button type="submit" disabled={pending || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
