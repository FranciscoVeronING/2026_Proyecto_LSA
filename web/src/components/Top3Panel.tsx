"use client";

import type { Top3Item } from "@/lib/schemas";

type Props = {
  title?: string;
  top3: Top3Item[];
  pendingGlosses: string;
  threshold?: number;
  emptyHint?: string;
};

export function Top3Panel({
  title = "Top 3 señas",
  top3,
  pendingGlosses,
  threshold = 0.75,
  emptyHint = "Esperando seña…",
}: Props) {
  return (
    <div className="rounded-xl bg-surface p-4 space-y-3 h-full">
      <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wide">
        {title}
      </h3>
      {top3.length === 0 ? (
        <p className="text-gray-500 text-sm">{emptyHint}</p>
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
        <p className="text-xs text-gray-400 mb-1">Buffer de glosas</p>
        <p className="text-cyan-300 text-sm font-mono min-h-[1.25rem]">
          {pendingGlosses || "(vacío)"}
        </p>
      </div>
    </div>
  );
}
