"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import {
  Activity,
  Brain,
  Map,
  Satellite,
  ShieldAlert,
} from "lucide-react";

const navigation = [
  { label: "Overview", icon: Activity },
  { label: "Satellite Intelligence", icon: Satellite },
  { label: "Risk Forecast", icon: Map },
  { label: "Distribution Shift", icon: ShieldAlert },
  { label: "AI Analysis", icon: Brain },
];

const metrics = [
  ["Active Anomalies", "—"],
  ["Risk Status", "—"],
  ["Model Confidence", "—"],
  ["Shift Status", "—"],
];

type Intelligence = {
  latitude: number;
  longitude: number;
  satellite_status: string;
  environmental_status: string;
  anomaly_status: string;
  risk_status: string;
};

const EarthMap = dynamic(() => import("@/components/earth-map"), {
  ssr: false,
});

export default function Home() {
  const [position, setPosition] = useState<[number, number] | null>(null);
  const [intelligence, setIntelligence] = useState<Intelligence | null>(null);

  const selectRegion = async (position: [number, number]) => {
    setPosition(position);

    const response = await fetch("http://127.0.0.1:8000/api/region", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: position[0],
        longitude: position[1],
      }),
    });

    const data: Intelligence = await response.json();
    setIntelligence(data);
  };

  return (
    <main className="flex min-h-screen bg-background text-foreground">
      <aside className="w-64 shrink-0 border-r p-6">
        <h1 className="text-xl font-semibold">EarthShield AI</h1>

        <p className="mt-1 text-xs text-muted-foreground">
          Multimodal Earth Intelligence
        </p>

        <nav className="mt-8 space-y-1">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm hover:bg-muted"
            >
              <Icon size={17} />
              {label}
            </button>
          ))}
        </nav>
      </aside>

      <section className="flex-1 p-8">
        <header>
          <p className="text-sm text-muted-foreground">SYSTEM OVERVIEW</p>

          <h2 className="mt-1 text-2xl font-semibold">
            Earth Intelligence Dashboard
          </h2>
        </header>

        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map(([label, value]) => (
            <div key={label} className="rounded-lg border p-5">
              <p className="text-sm text-muted-foreground">{label}</p>
              <p className="mt-3 text-2xl font-semibold">{value}</p>
            </div>
          ))}
        </div>

        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_320px]">
          <div className="rounded-lg border p-6">
            <p className="text-sm text-muted-foreground">
              EARTH OBSERVATION MAP
            </p>

            <div className="mt-4 h-[500px] overflow-hidden rounded-md">
              <EarthMap
                position={position}
                onSelect={selectRegion}
              />
            </div>
          </div>

          <div className="rounded-lg border p-6">
            <p className="text-sm text-muted-foreground">
              CURRENT INTELLIGENCE
            </p>

            <div className="mt-6 space-y-5 text-sm">
              <div>
                <p className="text-muted-foreground">Selected Region</p>
                <p className="mt-1">
                  {intelligence
                    ? `${intelligence.latitude.toFixed(4)}, ${intelligence.longitude.toFixed(4)}`
                    : "No region selected"}
                </p>
              </div>

              <div>
                <p className="text-muted-foreground">Satellite</p>
                <p className="mt-1">
                  {intelligence?.satellite_status ?? "Awaiting selection"}
                </p>
              </div>

              <div>
                <p className="text-muted-foreground">
                  Environmental Data
                </p>
                <p className="mt-1">
                  {intelligence?.environmental_status ??
                    "Awaiting selection"}
                </p>
              </div>

              <div>
                <p className="text-muted-foreground">Anomaly</p>
                <p className="mt-1">
                  {intelligence?.anomaly_status ?? "Awaiting selection"}
                </p>
              </div>

              <div>
                <p className="text-muted-foreground">Risk Forecast</p>
                <p className="mt-1">
                  {intelligence?.risk_status ?? "Awaiting selection"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}