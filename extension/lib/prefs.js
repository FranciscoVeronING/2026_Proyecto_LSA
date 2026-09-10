/**
 * Preferencias del popup (chrome.storage.local).
 * `outputMode`: subtitles | audio | both
 *
 * En offscreen a veces no hay `chrome.storage`; nunca hay que tirar el script.
 */

const LSA_PREFS_KEY = "lsaPrefs";

const LSA_PREFS_DEFAULTS = {
  showLandmarks: false,
  outputMode: "subtitles",
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
 * @param {object} [raw]
 * @returns {{ showLandmarks: boolean, outputMode: string }}
 */
function normalizeLsaPrefs(raw) {
  const src = raw && typeof raw === "object" ? raw : {};
  const mode = src.outputMode;
  return {
    showLandmarks: Boolean(src.showLandmarks),
    outputMode: mode === "audio" || mode === "both" ? mode : "subtitles",
  };
}

const LsaPrefs = {
  /**
   * @returns {Promise<{ showLandmarks: boolean, outputMode: string }>}
   */
  get() {
    const store = lsaStorageLocal();
    if (!store) return Promise.resolve({ ...LSA_PREFS_DEFAULTS });
    return store
      .get(LSA_PREFS_KEY)
      .then((data) => normalizeLsaPrefs((data && data[LSA_PREFS_KEY]) || LSA_PREFS_DEFAULTS))
      .catch(() => ({ ...LSA_PREFS_DEFAULTS }));
  },

  /**
   * @param {object} partial
   * @returns {Promise<{ showLandmarks: boolean, outputMode: string }>}
   */
  set(partial) {
    const store = lsaStorageLocal();
    return LsaPrefs.get().then((cur) => {
      const next = normalizeLsaPrefs({ ...cur, ...partial });
      if (!store) return next;
      return store.set({ [LSA_PREFS_KEY]: next }).then(() => next);
    });
  },

  /**
   * @param {(prefs: { showLandmarks: boolean, outputMode: string }) => void} fn
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
