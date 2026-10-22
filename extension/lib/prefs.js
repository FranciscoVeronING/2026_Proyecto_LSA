/**
 * Preferencias del popup (chrome.storage.local).
 * `outputMode`: subtitles | audio | both
 * `hearingInput`: voice | text | both  (oyente: mic de Meet, chat, o los dos)
 * `utterancePauseSec`: 2 | 3 | 4
 *
 * En offscreen a veces no hay `chrome.storage`; nunca hay que tirar el script.
 */

const LSA_PREFS_KEY = "lsaPrefs";

const LSA_PREFS_DEFAULTS = {
  showLandmarks: false,
  outputMode: "subtitles",
  hearingInput: "both",
  utterancePauseSec: 4,
  hudLeft: null,
  hudTop: null,
};

/**
 * @returns {chrome.storage.StorageArea|null}
 */
function lsaStorageLocal() {
  try {
    if (typeof chrome === "undefined" || !chrome.storage || !chrome.storage.local) {
      return null;
    }
    return chrome.storage.local;
  } catch (_) {
    return null;
  }
}

/**
 * @param {unknown} value
 * @returns {2|3|4}
 */
function normalizePauseSec(value) {
  const n = Number(value);
  if (n === 2 || n === 3 || n === 4) return n;
  return 4;
}

/**
 * @param {unknown} value
 * @returns {number|null}
 */
function normalizeHudPos(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.round(n);
}
function normalizeHearingInput(value) {
  if (value === "voice" || value === "text" || value === "both") return value;
  return "both";
}

/**
 * @returns {{ showLandmarks: boolean, outputMode: string, hearingInput: string, utterancePauseSec: number }}
 */
function normalizeLsaPrefs(raw) {
  const src = raw && typeof raw === "object" ? raw : {};
  const mode = src.outputMode;
  return {
    showLandmarks: Boolean(src.showLandmarks),
    outputMode: mode === "audio" || mode === "both" ? mode : "subtitles",
    hearingInput: normalizeHearingInput(src.hearingInput),
    utterancePauseSec: normalizePauseSec(src.utterancePauseSec),
    hudLeft: normalizeHudPos(src.hudLeft),
    hudTop: normalizeHudPos(src.hudTop),
  };
}

const LsaPrefs = {
  /**
   * @returns {Promise<{ showLandmarks: boolean, outputMode: string, utterancePauseSec: number }>}
   */
  get() {
    try {
      const store = lsaStorageLocal();
      if (!store) return Promise.resolve({ ...LSA_PREFS_DEFAULTS });
      return Promise.resolve(store.get(LSA_PREFS_KEY))
        .then((data) => normalizeLsaPrefs((data && data[LSA_PREFS_KEY]) || LSA_PREFS_DEFAULTS))
        .catch(() => ({ ...LSA_PREFS_DEFAULTS }));
    } catch (_) {
      return Promise.resolve({ ...LSA_PREFS_DEFAULTS });
    }
  },

  /**
   * @param {object} partial
   * @returns {Promise<{ showLandmarks: boolean, outputMode: string, utterancePauseSec: number }>}
   */
  set(partial) {
    try {
      const store = lsaStorageLocal();
      return LsaPrefs.get().then((cur) => {
        const next = normalizeLsaPrefs({ ...cur, ...partial });
        if (!store) return next;
        return Promise.resolve(store.set({ [LSA_PREFS_KEY]: next }))
          .then(() => next)
          .catch(() => next);
      });
    } catch (_) {
      return Promise.resolve(normalizeLsaPrefs({ ...LSA_PREFS_DEFAULTS, ...partial }));
    }
  },

  /**
   * @param {(prefs: { showLandmarks: boolean, outputMode: string, utterancePauseSec: number }) => void} fn
   */
  onChange(fn) {
    const store = lsaStorageLocal();
    if (!store || !chrome.storage.onChanged) return;
    chrome.storage.onChanged.addListener((changes, area) => {
      if (area !== "local" || !changes[LSA_PREFS_KEY]) return;
      fn(normalizeLsaPrefs(changes[LSA_PREFS_KEY].newValue));
    });
  },
};
