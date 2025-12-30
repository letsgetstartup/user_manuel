import Link from "next/link";
import { ArrowRight, BookOpen, Wrench } from "lucide-react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-6 bg-gradient-to-b from-background to-secondary/20">
      <div className="w-full max-w-md text-center space-y-6">
        <div className="mx-auto w-16 h-16 bg-primary/20 rounded-2xl flex items-center justify-center mb-6 ring-1 ring-primary/50 shadow-[0_0_30px_-10px_var(--primary)]">
          <Wrench className="w-8 h-8 text-primary" />
        </div>

        <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-br from-white to-gray-400 bg-clip-text text-transparent">
          Universal Guide AI
        </h1>
        <p className="text-muted-foreground text-lg">
          Your intelligent assistant for technical troubleshooting and maintenance.
        </p>

        <div className="grid gap-4 w-full mt-10">
          <Link href="/chat?machine=bes875-instruction-manual"
            className="group relative flex items-center p-4 bg-card hover:bg-accent/50 transition-all rounded-xl border border-border hover:border-primary/50 shadow-sm hover:shadow-md overflow-hidden">

            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

            <div className="bg-primary/20 p-3 rounded-lg mr-4">
              <BookOpen className="w-6 h-6 text-primary" />
            </div>

            <div className="text-left flex-1">
              <h3 className="font-semibold text-foreground">Sage Barista Express</h3>
              <p className="text-sm text-muted-foreground">Instruction Manual</p>
            </div>

            <ArrowRight className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
          </Link>

          <div className="p-4 rounded-xl border border-dashed border-muted bg-muted/20 text-center">
            <p className="text-sm text-muted-foreground">More manuals coming soon...</p>
          </div>
        </div>
      </div>
    </main>
  );
}
