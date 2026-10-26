export async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    signal: options.signal || AbortSignal.timeout(15000),
  });
  let result;
  try {
    result = await response.json();
  } catch {
    throw new Error(
      "The local detection service returned an unreadable response.",
    );
  }
  if (!response.ok) {
    const detail = result.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail
              .map((e) => `${e.loc?.slice(1).join(".")}: ${e.msg}`)
              .join("; ")
          : "Request failed. Try again.",
    );
  }
  return result;
}
export const defaults = () => ({
  upper: {
    camera: 0,
    confidence: 0.35,
    imgsz: 960,
    roi: [0, 0, 1, 1],
    sahi_wrist: true,
    sahi_slice: 640,
  },
  lower: {
    camera: 1,
    confidence: 0.35,
    imgsz: 960,
    roi: [0, 0, 1, 1],
    sahi_wrist: false,
    sahi_slice: 640,
  },
  profiles: {},
});
