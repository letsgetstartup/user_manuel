export const API_BASE_URL = "/api";

export interface ChatMessage {
    role: "user" | "model" | "assistant";
    content: string;
    data?: any; // For graph nodes or other structured data
}

export interface GraphNode {
    id: string;
    label: string;
    text: string;
    type: string;
    ar_anchor?: string;
    next_possible_steps?: GraphEdge[];
}

export interface GraphEdge {
    from: string;
    to: string;
    condition: string;
}

export async function searchGraph(machineId: string, query: string) {
    const res = await fetch(`${API_BASE_URL}/search_graph`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ machine_id: machineId, query }),
    });
    if (!res.ok) throw new Error("Graph search failed");
    return res.json();
}

export async function analyzeMessage(message: string, history: any[], machineId?: string | null, image?: File) {
    const formData = new FormData();
    formData.append("message", message);
    formData.append("history", JSON.stringify(history));
    if (machineId) {
        formData.append("machine_id", machineId);
    }
    if (image) {
        formData.append("image", image);
    }

    const res = await fetch(`${API_BASE_URL}/analyze`, {
        method: "POST",
        body: formData,
    });

    if (!res.ok) throw new Error("Analysis failed");
    return res.json();
}
