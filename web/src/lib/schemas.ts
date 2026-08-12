import { z } from "zod";

export const Top3ItemSchema = z.object({
  name: z.string(),
  confidence: z.number(),
});

export const ServerMessageSchema = z.object({
  type: z.string(),
  payload: z.record(z.unknown()).default({}),
});

export const ChatMessageSchema = z.object({
  participant_id: z.string(),
  participant_name: z.string().optional(),
  text: z.string(),
  source: z.enum(["typed", "stt", "interpretation"]).optional(),
  glosses: z.string().optional(),
  is_signer: z.boolean().optional(),
});

export const SessionSettingsSchema = z.object({
  confidence_threshold: z.number().min(0.1).max(1).default(0.75),
  motion_pixel_threshold: z.number().min(100).max(5000).default(500),
  still_frames_limit: z.number().min(5).max(40).default(10),
  static_hands_frames_to_start: z.number().min(2).max(15).default(4),
  capture_mode: z.enum(["auto", "static", "dynamic"]).default("auto"),
  utterance_pause_sec: z.number().default(4),
  letter_max_consecutive: z.number().default(2),
  voice_enabled: z.boolean().default(true),
  show_landmarks: z.boolean().default(false),
});

export type SessionSettings = z.infer<typeof SessionSettingsSchema>;
export type ChatMessage = z.infer<typeof ChatMessageSchema>;
export type Top3Item = z.infer<typeof Top3ItemSchema>;

export const DEFAULT_SETTINGS: SessionSettings = SessionSettingsSchema.parse({});

export type InterpretationMode = "voice" | "text" | "both";

export type PeerInfo = {
  participant_id: string;
  name: string;
  is_signer: boolean;
  left_handed: boolean;
};
