"use client";

import { Suspense } from "react";
import { useSearchParams, useParams } from "next/navigation";
import { CallRoom } from "@/components/CallRoom";

function CallContent() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const roomId = params.id;
  const name = searchParams.get("name") || "Anónimo";
  const isSigner = searchParams.get("isSigner") === "true";
  const leftHanded = searchParams.get("leftHanded") === "true";

  return (
    <CallRoom
      roomId={roomId}
      name={name}
      isSigner={isSigner}
      leftHanded={leftHanded}
    />
  );
}

export default function CallPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center">Cargando llamada…</div>}>
      <CallContent />
    </Suspense>
  );
}
