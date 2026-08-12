import { JoinForm } from "@/components/JoinForm";

type Props = {
  params: Promise<{ id: string }>;
};

export default async function RoomPage({ params }: Props) {
  const { id } = await params;
  return (
    <main>
      <JoinForm roomId={id} />
    </main>
  );
}
