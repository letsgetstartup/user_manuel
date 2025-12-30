"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { ArrowLeft, Send, Image as ImageIcon, Loader2, Sparkles, AlertCircle } from "lucide-react";
import Link from "next/link";
import { analyzeMessage, GraphNode, searchGraph } from "@/lib/api";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
    role: "user" | "assistant" | "model";
    content: string;
    data?: any; // For structured responses
}

function ChatContent() {
    const searchParams = useSearchParams();
    const machineId = searchParams.get("machine");
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initial Greeting
        if (messages.length === 0) {
            setMessages([
                {
                    role: "assistant",
                    content: "Hello! I'm your AI technical support guide. How can I help you with your machine today?",
                },
            ]);
        }
    }, []);

    useEffect(() => {
        // Scroll to bottom
        if (scrollRef.current) {
            scrollRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [messages]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || isLoading) return;

        const userMsg: Message = { role: "user", content: input };
        setMessages((prev) => [...prev, userMsg]);
        setInput("");
        setIsLoading(true);

        try {
            // 1. Parallel: Search Graph & General Analysis
            // ideally we do graph search first for specific solutions
            let graphResult = null;
            if (machineId) {
                try {
                    const res = await searchGraph(machineId, userMsg.content);
                    graphResult = res.result;
                } catch (err) {
                    console.warn("Graph search failed", err);
                }
            }

            if (graphResult) {
                setMessages(prev => [...prev, {
                    role: "assistant",
                    content: `I found a relevant solution step: ${graphResult.label}`,
                    data: { type: 'graph_node', ...graphResult }
                }]);
            } else {
                // Fallback to Gemini
                const response = await analyzeMessage(userMsg.content, messages, machineId);
                if (response.solution) {
                    setMessages((prev) => [...prev, { role: "assistant", content: response.solution }]);
                } else if (response.tutorial) {
                    setMessages((prev) => [
                        ...prev,
                        { role: "assistant", content: "Here is a step-by-step tutorial:", data: { type: 'tutorial', ...response.tutorial } },
                    ]);
                }
            }

        } catch (error) {
            console.error(error);
            setMessages((prev) => [
                ...prev,
                { role: "assistant", content: "Sorry, I encountered an error. Please try again." },
            ]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen max-w-lg mx-auto bg-background shadow-2xl overflow-hidden relative">
            {/* Header */}
            <header className="flex items-center p-4 border-b border-border bg-background/80 backdrop-blur-md sticky top-0 z-10">
                <Link href="/" className="p-2 hover:bg-accent rounded-full transition-colors">
                    <ArrowLeft className="w-5 h-5" />
                </Link>
                <div className="ml-3">
                    <h1 className="font-semibold text-sm">Technical Support</h1>
                    <p className="text-xs text-muted-foreground">{machineId || "Universal AI"}</p>
                </div>
            </header>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-6">
                {messages.map((msg, idx) => (
                    <ChatMessage key={idx} message={msg} />
                ))}
                {isLoading && (
                    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex items-center gap-2 text-muted-foreground text-sm ml-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>Analyzing...</span>
                    </motion.div>
                )}
                <div ref={scrollRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-border bg-background safe-area-inset-bottom">
                <form onSubmit={handleSubmit} className="flex items-end gap-2 bg-secondary/50 p-2 rounded-2xl ring-1 ring-border focus-within:ring-primary/50 focus-within:bg-secondary transition-all">
                    <button type="button" className="p-2 text-muted-foreground hover:text-foreground transition-colors">
                        <ImageIcon className="w-5 h-5" />
                    </button>
                    <textarea
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="Ask a question..."
                        className="flex-1 bg-transparent border-0 focus:ring-0 resize-none max-h-32 min-h-[40px] py-2 px-1 text-sm outline-none"
                        rows={1}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit(e);
                            }
                        }}
                    />
                    <button
                        type="submit"
                        disabled={!input.trim() || isLoading}
                        className="p-2 bg-primary text-primary-foreground rounded-xl disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
                    >
                        <Send className="w-5 h-5" />
                    </button>
                </form>
            </div>
        </div>
    );
}

function ChatMessage({ message }: { message: Message }) {
    const isUser = message.role === "user";

    return (
        <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={cn("flex w-full", isUser ? "justify-end" : "justify-start")}
        >
            <div
                className={cn(
                    "max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed",
                    isUser
                        ? "bg-primary text-primary-foreground rounded-br-none"
                        : "bg-secondary text-secondary-foreground rounded-bl-none"
                )}
            >
                <div className="whitespace-pre-wrap">{message.content}</div>

                {/* Structured Data Renderer */}
                {message.data?.type === "graph_node" && (
                    <div className="mt-4 p-3 bg-background/50 rounded-xl border border-white/10">
                        <div className="flex items-center gap-2 mb-2 text-primary font-semibold">
                            <Sparkles className="w-4 h-4" />
                            <span>Suggested Action</span>
                        </div>
                        <h4 className="font-bold mb-1">{message.data.label}</h4>
                        <p className="text-muted-foreground mb-3">{message.data.text}</p>

                        {message.data.ar_anchor && (
                            <div className="text-xs bg-blue-500/10 text-blue-400 px-2 py-1 rounded inline-block mb-3">
                                📍 Check: {message.data.ar_anchor}
                            </div>
                        )}

                        {message.data.next_possible_steps?.length > 0 && (
                            <div className="space-y-2 mt-2 pt-2 border-t border-white/5">
                                <p className="text-xs uppercase tracking-wider text-muted-foreground font-bold">Try next:</p>
                                {message.data.next_possible_steps.map((step: any, i: number) => (
                                    <div key={i} className="text-xs p-2 bg-background/80 rounded border border-white/5">
                                        If <span className="text-accent-foreground font-medium">{step.condition}</span> → {step.to}
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {message.data?.type === "tutorial" && (
                    <div className="mt-4 space-y-4">
                        {message.data.steps.map((step: any) => (
                            <div key={step.step_number} className="bg-background/40 p-3 rounded-lg">
                                <div className="flex justify-between items-start mb-2">
                                    <span className="font-mono text-xs text-muted-foreground">step {step.step_number}</span>
                                    {step.pdf_page_reference && (
                                        <span className="text-[10px] bg-white/10 px-1.5 py-0.5 rounded">p.{step.pdf_page_reference}</span>
                                    )}
                                </div>
                                <p className="font-medium mb-2">{step.instruction}</p>
                                {step.image_url ? (
                                    <img src={step.image_url} alt="Step visual" className="rounded-lg w-full h-auto border border-white/10" />
                                ) : step.has_visual ? (
                                    <div className="flex items-center gap-2 text-xs text-yellow-400 bg-yellow-400/10 p-2 rounded">
                                        <AlertCircle className="w-3 h-3" />
                                        <span>Visual available in manual (p.{step.pdf_page_reference})</span>
                                    </div>
                                ) : null}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </motion.div>
    );
}

export default function Page() {
    return (
        <React.Suspense fallback={<div className="flex h-screen items-center justify-center"><Loader2 className="animate-spin" /></div>}>
            <ChatContent />
        </React.Suspense>
    )
}
import React from "react";
