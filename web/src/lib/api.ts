const DEFAULT_BACKEND_PORT = 8000;

function configuredApiOrigin(): string {
  return process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "";
}

/** Host para WebSocket. Con Tailscale Serve HTTPS usa el mismo host que la página. */
export function backendHost(): string {
  if (typeof window !== "undefined") {
    const pageHost = window.location.host;
    const apiOrigin = configuredApiOrigin();
    if (!apiOrigin || apiOrigin === window.location.origin) {
      return pageHost;
    }
  }

  const apiUrl = configuredApiOrigin();
  if (apiUrl) {
    try {
      return new URL(apiUrl).host;
    } catch {
      /* ignore */
    }
  }

  const raw = process.env.NEXT_PUBLIC_API_HOST || `localhost:${DEFAULT_BACKEND_PORT}`;
  if (/:\d+$/.test(raw)) return raw;
  return `${raw}:${DEFAULT_BACKEND_PORT}`;
}

function backendHttpBase(): string {
  if (typeof window !== "undefined") {
    const apiOrigin = configuredApiOrigin();
    if (!apiOrigin || apiOrigin === window.location.origin) return "";
  }

  const fromEnv = configuredApiOrigin();
  if (!fromEnv) return "";
  try {
    const u = new URL(fromEnv);
    if (u.protocol === "https:" && !u.port) return fromEnv;
    if (u.port) return fromEnv;
  } catch {
    /* ignore */
  }
  if (/:\d+/.test(fromEnv.replace(/^https?:\/\//, ""))) return fromEnv;
  return `${fromEnv}:${DEFAULT_BACKEND_PORT}`;
}

function apiUrl(path: string): string {
  const base = backendHttpBase();
  return base ? `${base}${path}` : path;
}

export async function createRoom(): Promise<string> {
  const res = await fetch(apiUrl("/api/rooms"), { method: "POST" });
  if (!res.ok) throw new Error("No se pudo crear la sala");
  const data = await res.json();
  return data.room_id as string;
}

export async function getRoomStatus(roomId: string) {
  const res = await fetch(apiUrl(`/api/rooms/${roomId}`));
  if (!res.ok) return null;
  return res.json();
}

export function wsUrl(roomId: string): string {
  const base = process.env.NEXT_PUBLIC_WS_URL;
  if (base) return `${base}/ws/${roomId}`;
  if (typeof window === "undefined") return "";
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${backendHost()}/ws/${roomId}`;
}
