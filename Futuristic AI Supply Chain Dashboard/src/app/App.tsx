import React, { useEffect, useRef, useState, useMemo } from "react";
import { LineChart, Box, Network, Shuffle, Repeat, FileText, Bell, HelpCircle } from "lucide-react";

// ── Icons ─────────────────────────────────────────────────────────────────
function ModelRepositoryIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="#00aaff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ filter: "drop-shadow(0px 0px 6px rgba(0,170,255,0.8))" }}>
      {/* Database */}
      <ellipse cx="32" cy="32" rx="8" ry="3" />
      <path d="M24 32v8c0 1.66 3.58 3 8 3s8-1.34 8-3v-8" />
      <path d="M24 36c0 1.66 3.58 3 8 3s8-1.34 8-3" />
      {/* Top Left Doc */}
      <rect x="12" y="14" width="10" height="12" rx="1" />
      <path d="M15 18h4 M15 22h4" />
      {/* Top Right Doc */}
      <rect x="42" y="14" width="10" height="12" rx="1" />
      <path d="M45 18h4 M45 22h4" />
      {/* Bottom Left Doc */}
      <rect x="12" y="38" width="10" height="12" rx="1" />
      <path d="M15 42h4 M15 46h4" />
      {/* Bottom Right Gear */}
      <circle cx="47" cy="44" r="5" />
      <path d="M47 37v2 M47 49v2 M40 44h2 M52 44h2 M42 39l1.5 1.5 M50.5 47.5l1.5 1.5 M50.5 39l-1.5 1.5 M42 49l1.5-1.5" />
      {/* Arrows */}
      <path d="M24 20h8l2 9" />
      <path d="M40 20h-8l-2 9" />
      <path d="M24 44h8l2-9" />
    </svg>
  );
}

function SimulatorIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="#00aaff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ filter: "drop-shadow(0px 0px 6px rgba(0,170,255,0.8))" }}>
      {/* CPU Body */}
      <rect x="16" y="20" width="32" height="28" rx="3" />
      {/* Pins Top */}
      <path d="M22 20v-5 M32 20v-5 M42 20v-5" />
      <circle cx="22" cy="13" r="2" />
      <circle cx="32" cy="13" r="2" />
      <circle cx="42" cy="13" r="2" />
      {/* Pins Bottom */}
      <path d="M22 48v5 M32 48v5 M42 48v5" />
      <circle cx="22" cy="55" r="2" />
      <circle cx="32" cy="55" r="2" />
      <circle cx="42" cy="55" r="2" />
      {/* Pins Left */}
      <path d="M16 26h-4 M16 34h-4 M16 42h-4" />
      {/* Pins Right */}
      <path d="M48 26h4 M48 34h4 M48 42h4" />
      {/* Play Button */}
      <polygon points="28,28 28,40 38,34" fill="rgba(0,170,255,0.2)" />
    </svg>
  );
}

function StrategyIcon() {
  return (
    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="#00aaff" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ filter: "drop-shadow(0px 0px 6px rgba(0,170,255,0.8))" }}>
      {/* Nodes */}
      <circle cx="32" cy="50" r="4" fill="rgba(0,170,255,0.2)" />
      <circle cx="20" cy="34" r="4" />
      <circle cx="44" cy="34" r="4" />
      <circle cx="20" cy="18" r="4" />
      <circle cx="32" cy="18" r="4" />
      {/* Lines */}
      <path d="M30 46 l-8 -8 M34 46 l8 -8 M20 30 v-8 M22 30 l8 -8 M44 30 v-8" />
      {/* Big Arrow pointing Top Right */}
      <path d="M38 30 l12 -12" strokeWidth="2.5" />
      <polygon points="52,16 42,16 52,26" fill="#00aaff" />
    </svg>
  );
}

// ── Animated Wireframe Sphere ──────────────────────────────────────────────
function WireframeSphere() {
  const [angle, setAngle] = useState(0);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);

  useEffect(() => {
    const animate = (t: number) => {
      if (!startRef.current) startRef.current = t;
      setAngle(((t - startRef.current) / 10000) * Math.PI * 2);
      rafRef.current = requestAnimationFrame(animate);
    };
    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, []);

  const R = 320;
  const cx = 350;
  const cy = 350;

  const numNodes = 70;
  const nodes = useMemo(() => {
    const phi = Math.PI * (3 - Math.sqrt(5));
    return Array.from({ length: numNodes }, (_, i) => {
      const y = 1 - (i / (numNodes - 1)) * 2;
      const radiusAtY = Math.sqrt(1 - y * y);
      const theta = phi * i;
      return { 
        x: Math.cos(theta) * radiusAtY, 
        y, 
        z: Math.sin(theta) * radiusAtY, 
        key: i,
        hasBox: Math.random() > 0.85,
      };
    });
  }, []);

  const sphereEdges = useMemo(() => {
    const edges: [number, number][] = [];
    for (let i = 0; i < nodes.length; i++) {
      const distances = [];
      for (let j = 0; j < nodes.length; j++) {
        if (i !== j) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dz = nodes[i].z - nodes[j].z;
          distances.push({ index: j, dist: dx*dx + dy*dy + dz*dz });
        }
      }
      distances.sort((a, b) => a.dist - b.dist);
      for (let k = 0; k < 3; k++) {
        const j = distances[k].index;
        if (i < j) edges.push([i, j]);
      }
    }
    return edges;
  }, [nodes]);

  const projectedNodes = nodes.map(node => {
    const sinA = Math.sin(angle);
    const cosA = Math.cos(angle);
    const rx = node.x * cosA - node.z * sinA;
    const rz = node.x * sinA + node.z * cosA;
    
    const tilt = 0.2;
    const sinT = Math.sin(tilt);
    const cosT = Math.cos(tilt);
    const ry = node.y * cosT - rz * sinT;
    const rz2 = node.y * sinT + rz * cosT;

    return {
      px: cx + rx * R,
      py: cy + ry * R,
      z: rz2,
      vis: rz2 > -0.25,
      key: node.key,
      hasBox: node.hasBox
    };
  });

  const meridianCount = 14;
  const meridians = Array.from({ length: meridianCount }, (_, i) => {
    const phi = angle + (i * Math.PI) / meridianCount;
    const cosPhi = Math.cos(phi);
    const rx = Math.abs(cosPhi) * R;
    const opacity = ((cosPhi + 1.2) / 2.2) * 0.4;
    return { rx, opacity, key: i };
  });

  const latitudes = [-0.8, -0.6, -0.3, 0, 0.3, 0.6, 0.8].map((f) => ({
    rx: Math.sqrt(Math.max(0, 1 - f * f)) * R,
    ry: 16,
    y: cy + f * R,
    opacity: 0.1 + Math.abs(f) * 0.08,
  }));

  return (
    <svg viewBox={`0 0 ${cx * 2} ${cy * 2}`} className="w-full h-full overflow-visible">
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <filter id="strongGlow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="12" result="blur1" />
          <feGaussianBlur stdDeviation="4" result="blur2" />
          <feMerge>
            <feMergeNode in="blur1" />
            <feMergeNode in="blur2" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        <radialGradient id="sphereAmbient" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#00aaff" stopOpacity="0.12" />
          <stop offset="60%" stopColor="#0044aa" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#00aaff" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Ambient backgrounds */}
      <circle cx={cx} cy={cy} r={R + 60} fill="url(#sphereAmbient)" />
      
      {/* Outer borders */}
      <circle cx={cx} cy={cy} r={R} fill="none" stroke="#00aaff" strokeWidth="2.5" opacity="0.6" filter="url(#strongGlow)" />
      <circle cx={cx} cy={cy} r={R + 15} fill="none" stroke="#00aaff" strokeWidth="1" opacity="0.2" />
      <circle cx={cx} cy={cy} r={R + 35} fill="none" stroke="#00aaff" strokeWidth="1" opacity="0.1" strokeDasharray="6 8" />

      {/* Sphere Lat/Lon base */}
      <g opacity="0.6">
        <clipPath id="sphereClip">
          <circle cx={cx} cy={cy} r={R} />
        </clipPath>
        {latitudes.map(({ rx, ry, y, opacity }, i) => (
          <ellipse key={`lat-${i}`} cx={cx} cy={y} rx={rx} ry={ry} fill="none" stroke="#00aaff" strokeWidth="1" opacity={opacity} />
        ))}
        {meridians.map(({ rx, opacity, key }) => (
          <ellipse key={`mer-${key}`} cx={cx} cy={cy} rx={rx} ry={R} fill="none" stroke="#00aaff" strokeWidth="1" opacity={opacity} clipPath="url(#sphereClip)" />
        ))}
      </g>

      {/* Network Edges */}
      {sphereEdges.map(([i, j], idx) => {
        const n1 = projectedNodes[i];
        const n2 = projectedNodes[j];
        if (n1.vis && n2.vis) {
          const zDepth = (n1.z + n2.z) / 2;
          const opacity = Math.min(1, Math.max(0, zDepth + 0.6)) * 0.8;
          return (
             <line key={`e-${idx}`} x1={n1.px} y1={n1.py} x2={n2.px} y2={n2.py} stroke="#00aaff" strokeWidth={zDepth > 0 ? "1.2" : "0.5"} opacity={opacity} filter={zDepth > 0 ? "url(#glow)" : undefined} />
          )
        }
        return null;
      })}

      {/* Network Nodes */}
      {projectedNodes.map((n) => 
        n.vis && (
          <g key={`n-${n.key}`}>
            <circle cx={n.px} cy={n.py} r={n.z > 0 ? 3.5 : 2} fill="#ffffff" filter="url(#strongGlow)" opacity={0.9 + n.z*0.1} />
            <circle cx={n.px} cy={n.py} r={n.z > 0 ? 8 : 4} fill="none" stroke="#00aaff" strokeWidth="1.5" opacity={0.6 + n.z*0.4} />
            
            {/* Small floating UI elements on some nodes */}
            {n.hasBox && n.z > 0.2 && (
               <g transform={`translate(${n.px + 12}, ${n.py - 12})`}>
                 <rect x="0" y="0" width="16" height="12" fill="rgba(0,170,255,0.15)" stroke="#00aaff" strokeWidth="0.8" rx="2" filter="url(#glow)" />
                 <line x1="3" y1="4" x2="13" y2="4" stroke="#00aaff" strokeWidth="1" opacity="0.8" />
                 <line x1="3" y1="8" x2="9" y2="8" stroke="#00aaff" strokeWidth="1" opacity="0.6" />
               </g>
            )}
          </g>
        )
      )}
    </svg>
  );
}

// ── UI Components ─────────────────────────────────────────────────────────

function Card({ icon, title }: { icon: React.ReactNode, title: React.ReactNode }) {
  return (
    <div className="w-[180px] h-[190px] flex flex-col items-center justify-center p-4 rounded-[1.25rem] relative overflow-hidden transition-transform duration-300 hover:-translate-y-1 cursor-pointer group"
      style={{
        background: "linear-gradient(180deg, rgba(16, 42, 80, 0.4) 0%, rgba(8, 22, 48, 0.2) 100%)",
        backdropFilter: "blur(16px)",
        border: "1px solid rgba(0, 170, 255, 0.2)",
        boxShadow: "inset 0 0 20px rgba(0,170,255,0.05), 0 8px 32px rgba(0,0,0,0.3)"
      }}
    >
      {/* Top glowing edge */}
      <div className="absolute top-0 left-1/4 right-1/4 h-[1px] bg-gradient-to-r from-transparent via-[#00aaff] to-transparent opacity-60 group-hover:opacity-100 transition-opacity" />
      <div className="absolute top-0 left-1/4 right-1/4 h-[10px] bg-gradient-to-b from-[#00aaff] to-transparent opacity-10 blur-md group-hover:opacity-20 transition-opacity" />
      
      <div className="flex-1 flex items-center justify-center pt-2">
        {icon}
      </div>
      <div className="mt-auto h-[60px] flex items-center justify-center text-center">
        <h3 className="text-[#e8f0fe] text-sm font-medium leading-relaxed tracking-wide">{title}</h3>
      </div>
    </div>
  )
}

function Pill({ icon, label }: { icon: React.ReactNode, label: string }) {
  return (
    <div className="flex items-center gap-2.5 px-4 py-2 rounded-full cursor-pointer hover:bg-[#00aaff]/10 transition-colors"
      style={{
        background: "rgba(10, 30, 60, 0.5)",
        border: "1px solid rgba(0, 170, 255, 0.2)",
        backdropFilter: "blur(8px)"
      }}
    >
      <div className="text-[#00aaff]">
        {icon}
      </div>
      <span className="text-[#a5d8ff] text-xs font-medium tracking-wide">{label}</span>
    </div>
  )
}

// ── Main App ───────────────────────────────────────────────────────────────
export default function App() {
  return (
    <div className="min-h-screen w-full flex flex-col font-sans overflow-hidden" style={{ background: "#040d1c" }}>
      {/* Grid background */}
      <div 
        className="fixed inset-0 pointer-events-none"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0, 170, 255, 0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 170, 255, 0.04) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
          backgroundPosition: "center center",
        }}
      />

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-10 py-5 border-b border-[#00aaff]/10 bg-[#040d1c]/80 backdrop-blur-md relative z-10">
        <div className="flex items-center gap-12">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00aaff" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ filter: "drop-shadow(0 0 8px rgba(0,170,255,0.6))" }}>
              <path d="M12 2 a10 10 0 0 0 -10 10 a10 10 0 0 0 10 10" />
              <circle cx="12" cy="2" r="2.5" fill="#040d1c" />
              <circle cx="2" cy="12" r="2.5" fill="#040d1c" />
              <circle cx="12" cy="22" r="2.5" fill="#040d1c" />
              <circle cx="7" cy="12" r="1.5" fill="#00aaff" />
            </svg>
            <span className="text-xl font-bold tracking-wide bg-gradient-to-r from-[#22d3ff] via-[#38bdf8] to-[#3b82f6] bg-clip-text text-transparent" style={{ filter: "drop-shadow(0 0 8px rgba(34,211,255,0.6))" }}>ChainSight</span>
          </div>

          {/* Nav Links */}
          <nav className="hidden lg:flex items-center gap-8 ml-8">
            {["Dashboard", "Analytics", "Simulation", "Reports", "Settings"].map((item, i) => (
              <a key={item} href="#" className={`text-[15px] font-medium ${i === 0 ? "text-[#00aaff] relative" : "text-[#7b98b0] hover:text-[#a5d8ff]"} transition-colors`}>
                {item}
                {i === 0 && (
                  <div className="absolute -bottom-[23px] left-0 right-0 h-[2px] bg-[#00aaff] shadow-[0_0_12px_#00aaff]" />
                )}
              </a>
            ))}
          </nav>
        </div>

        {/* Right Icons */}
        <div className="flex items-center gap-6">
          <div className="relative cursor-pointer group">
            <Bell size={22} className="text-[#7b98b0] group-hover:text-white transition-colors" />
            <span className="absolute top-0 right-0 w-2 h-2 bg-red-500 rounded-full border-[1.5px] border-[#040d1c]" />
          </div>
          <div className="w-8 h-8 rounded-full overflow-hidden border border-[#00aaff]/40 cursor-pointer hover:border-[#00aaff] transition-colors">
            <img src="https://images.unsplash.com/photo-1500648767791-00dcc994a43e?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxwcm9mZXNzaW9uYWwlMjBwb3J0cmFpdCUyMHBob3RvJTIwYXZhdGFyfGVufDF8fHx8MTc4MjE5NjQwM3ww&ixlib=rb-4.1.0&q=80&w=108" alt="User Avatar" className="w-full h-full object-cover" />
          </div>
          <div className="cursor-pointer group">
            <HelpCircle size={22} className="text-[#7b98b0] group-hover:text-white transition-colors" />
          </div>
        </div>
      </header>

      {/* ── Main Content ── */}
      <main className="relative z-10 flex-1 grid grid-cols-1 lg:grid-cols-[1.1fr_0.9fr] max-w-[1800px] mx-auto w-full">
        
        {/* Left Panel */}
        <div className="flex flex-col justify-center px-10 lg:pl-20 py-12 lg:py-0">
          
          <div className="flex items-center mb-6">
            <h1 className="text-3xl md:text-4xl font-bold tracking-wide whitespace-nowrap" style={{ color: "#7ec8f0", textShadow: "0 0 24px rgba(126, 200, 240, 0.4)" }}>
              好计划 轻松做
            </h1>
            <div className="ml-6 h-[2px] w-32 md:w-64" style={{ background: "linear-gradient(90deg, #00aaff, transparent)", boxShadow: "0 0 16px #00aaff" }} />
          </div>
          
          <h2 className="text-2xl md:text-3xl font-medium text-white mb-12 leading-snug tracking-wide">
            <span className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-[#22d3ff] via-[#38bdf8] to-[#3b82f6] bg-clip-text text-transparent" style={{ filter: "drop-shadow(0 0 16px rgba(34,211,255,0.7))" }}>ChainSight</span><br/>Your Operating Strategy Simulator and Optimizer
          </h2>

          <div className="flex flex-wrap gap-5 mb-14">
            <Card 
              icon={<ModelRepositoryIcon />} 
              title={<>Model<br/>Repository</>} 
            />
            <Card 
              icon={<StrategyIcon />} 
              title={<>Strategy<br/>Recommendor</>} 
            />
            <Card 
              icon={<SimulatorIcon />} 
              title={<>Build Adhoc<br/>Simulator</>} 
            />
          </div>

          <div className="flex flex-wrap gap-3 max-w-[700px]">
            <Pill icon={<LineChart size={16}/>} label="Leveling Production" />
            <Pill icon={<Box size={16}/>} label="Space Optimization" />
            <Pill icon={<Network size={16}/>} label="Irinework E2E Simulation" />
            <Pill icon={<Shuffle size={16}/>} label="VS/Network E2E Simulation" />
            <Pill icon={<Repeat size={16}/>} label="Reverse Flow Simulation" />
            <Pill icon={<FileText size={16}/>} label="Inventory Target Simulation" />
          </div>
          
        </div>

        {/* Right Panel */}
        <div className="flex flex-col items-center justify-center relative min-h-[600px] pr-10">
          <div className="w-[500px] h-[500px] xl:w-[650px] xl:h-[650px] relative mt-10">
            <WireframeSphere />
          </div>
          <p className="text-[#89a8c0] text-[15px] mt-6 tracking-wide" style={{ textShadow: "0 2px 4px rgba(0,0,0,0.5)" }}>
            Supply Chain Insight &amp; Simulation &amp; Optimization
          </p>
        </div>
        
      </main>
    </div>
  );
}
