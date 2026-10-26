export function GogglesIcon({ className = "", size = 52 }) {
  return (
    <img
      src="/static/branding/goggles.png"
      width={size}
      height={size}
      className={`goggles-logo ${className}`}
      alt=""
      aria-hidden="true"
    />
  );
}

export function EarplugsIcon({ size = 52, color = "#64748b" }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" fill="none" aria-hidden="true">
      <g stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
        <path d="M17 31v12c0 10 30 10 30 0V31" />
        <path d="M12 10h10l3 18c.3 2-1 4-3 4H12c-2 0-3.3-2-3-4l3-18Z" />
        <path d="M42 10h10l3 18c.3 2-1 4-3 4H42c-2 0-3.3-2-3-4l3-18Z" />
        <path d="M12 18h10M42 18h10" />
      </g>
    </svg>
  );
}

// Safety shoe outline.
export function SafetyShoeIcon({
  className = "",
  size = 52,
  color = "#00a854",
}) {
  return (
    <svg
      width={size}
      height={size * 0.75}
      viewBox="0 0 100 75"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
    >
      {/* Boot Upper profile */}
      <path
        d="M24 10 L50 10 C53 10 55 13 54 18 L50 28 L74 34 C86 38 94 44 94 54 L94 59 C94 61 92 62 89 62 L16 62 C12 62 10 60 10 56 L10 18 C10 13 14 10 19 10 H24 Z"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Tread Sole with heel */}
      <path
        d="M9 62 L9 67 C9 70 11 71 14 71 L33 71 C36 71 37 69 37 67 L37 62"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M42 62 L46 67 C47 70 50 71 53 71 L88 71 C92 71 95 68 95 64 L95 62"
        stroke={color}
        strokeWidth="5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      {/* Boot laces stitches */}
      <line
        x1="48"
        y1="22"
        x2="55"
        y2="26"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
      <line
        x1="54"
        y1="29"
        x2="62"
        y2="33"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
      <line
        x1="61"
        y1="36"
        x2="69"
        y2="40"
        stroke={color}
        strokeWidth="4"
        strokeLinecap="round"
      />
    </svg>
  );
}

// PPE Detection System Shield Logo
export function ShieldCheckLogo({ size = 44 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none">
      <path
        d="M24 4 L6 11 C6 28 14 40 24 44 C34 40 42 28 42 11 L24 4 Z"
        fill="#00c853"
      />
      <path
        d="M16 23.5 L22 29.5 L32 18"
        stroke="#ffffff"
        strokeWidth="4.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export function BekaertLogo() {
  return (
    <img
      className="bekaert-logo"
      src="/static/branding/bekaert.jpg"
      width="3415"
      height="798"
      alt="Bekaert"
    />
  );
}
