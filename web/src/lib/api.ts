const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

export async function createRoom(): Promise<string> {
  const res = await fetch(`${API_BASE}/api/rooms`, { method: "POST" });
  if (!res.ok) throw new Error("No se pudo crear la sala");
  const data = await res.json();
  return data.room_id as string;
}

export async function getRoomStatus(roomId: string) {
  const res = await fetch(`${API_BASE}/api/rooms/${roomId}`);
  if (!res.ok) return null;
  return res.json();
}

export function wsUrl(roomId: string): string {
  const base = process.env.NEXT_PUBLIC_WS_URL;
  if (base) return `${base}/ws/${roomId}`;
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const host = process.env.NEXT_PUBLIC_API_HOST || "localhost:8000";
  return `${proto}//${host}/ws/${roomId}`;
}
