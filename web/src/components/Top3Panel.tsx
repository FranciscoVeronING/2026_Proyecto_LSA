"use client";

import type { Top3Item } from "@/lib/schemas";

type Props = {
  top3: Top3Item[];
  pendingGlosses: string;
  threshold?: number;
};

export function Top3Panel({ top3, pendingGlosses, threshold = 0.75 }: Props) {
  return (
    <div className="rounded-xl bg-surface p-4 space-y-3">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
        Top 3 señas
      </h3>
      {top3.length === 0 ? (
        <p className="text-gray-500 text-sm">Esperando seña…</p>
      ) : (
        <ul className="space-y-2">
          {top3.map((item, i) => (
            <li key={i} className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className={item.confidence >= threshold ? "text-green-400" : "text-yellow-400"}>
                  {item.name.toUpperCase()}
                </span>
                <span className="text-gray-400">{(item.confidence * 100).toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${item.confidence >= threshold ? "bg-green-500" : "bg-yellow-500"}`}
                  style={{ width: `${Math.min(item.confidence * 100, 100)}%` }}
                />
              </div>
            </li>
          ))}
        </ul>
      )}
      <div className="pt-2 border-t border-gray-700">
        <p className="text-xs text-gray-400 mb-1">Glosas en construcción</p>
        <p className="text-cyan-300 text-sm font-mono">
          {pendingGlosses || "(vacío)"}
        </p>
      </div>
    </div>
  );
}
