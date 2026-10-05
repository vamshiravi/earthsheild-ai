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

type Intelligence = {
  latitude: number;
  longitude: number;

  satellite_status: string;
  environmental_status: string;

  anomaly_status: string;
  risk_status: string;

  flood_risk: number;
  risk_level: string;

  anomaly_score: number;
  anomaly: number;

  shift_score: number;
  shift_ratio: number;
  shift_status: string;

  confidence_percent: number;
  earthshield_status: string;

  date: string;
};

const EarthMap = dynamic(() => import("@/components/earth-map"), {
  ssr: false,
});

export default function Home() {
  const [position, setPosition] = useState<[number, number] | null>(null);
  const [intelligence, setIntelligence] =
    useState<Intelligence | null>(null);
  const [loading, setLoading] = useState(false);

  const selectRegion = async (
    selectedPosition: [number, number]
  ) => {
    setPosition(selectedPosition);
    setLoading(true);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/region",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            latitude: selectedPosition[0],
            longitude: selectedPosition[1],
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to fetch region intelligence");
      }

      const data: Intelligence = await response.json();

      setIntelligence(data);
    } catch (error) {
      console.error(error);
      setIntelligence(null);
    } finally {
      setLoading(false);
    }
  };

  const metrics = [
    [
      "Risk Status",
      intelligence?.risk_level ?? "—",
    ],
    [
      "Anomaly",
      intelligence
        ? intelligence.anomaly_status
        : "—",
    ],
    [
      "Model Confidence",
      intelligence
        ? `${intelligence.confidence_percent.toFixed(0)}%`
        : "—",
    ],
    [
      "Shift Status",
      intelligence?.shift_status ?? "—",
    ],
  ];

  return (
    <main className="flex min-h-screen bg-background text-foreground">
      {/* SIDEBAR */}
      <aside className="w-64 shrink-0 border-r p-6">
        <h1 className="text-xl font-semibold">
          EarthShield AI
        </h1>

        <p className="mt-1 text-xs text-muted-foreground">
          Multimodal Earth Intelligence
        </p>

        <nav className="mt-8 space-y-1">
          {navigation.map(
            ({ label, icon: Icon }) => (
              <button
                key={label}
                className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm hover:bg-muted"
              >
                <Icon size={17} />
                {label}
              </button>
            )
          )}
        </nav>
      </aside>

      {/* MAIN */}
      <section className="flex-1 p-8">
        <header>
          <p className="text-sm text-muted-foreground">
            SYSTEM OVERVIEW
          </p>

          <h2 className="mt-1 text-2xl font-semibold">
            Earth Intelligence Dashboard
          </h2>
        </header>

        {/* METRICS */}
        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {metrics.map(([label, value]) => (
            <div
              key={label}
              className="rounded-lg border p-5"
            >
              <p className="text-sm text-muted-foreground">
                {label}
              </p>

              <p className="mt-3 text-2xl font-semibold capitalize">
                {loading ? "..." : value}
              </p>
            </div>
          ))}
        </div>

        {/* MAP + INTELLIGENCE */}
        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_320px]">
          {/* MAP */}
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

          {/* INTELLIGENCE */}
          <div className="rounded-lg border p-6">
            <p className="text-sm text-muted-foreground">
              CURRENT INTELLIGENCE
            </p>

            <div className="mt-6 space-y-5 text-sm">
              {/* REGION */}
              <div>
                <p className="text-muted-foreground">
                  Selected Region
                </p>

                <p className="mt-1">
                  {intelligence
                    ? `${intelligence.latitude.toFixed(
                        4
                      )}, ${intelligence.longitude.toFixed(
                        4
                      )}`
                    : "No region selected"}
                </p>
              </div>

              {/* DATE */}
              <div>
                <p className="text-muted-foreground">
                  Observation Date
                </p>

                <p className="mt-1">
                  {intelligence?.date ??
                    "Awaiting selection"}
                </p>
              </div>

              {/* SATELLITE */}
              <div>
                <p className="text-muted-foreground">
                  Satellite
                </p>

                <p className="mt-1">
                  {intelligence?.satellite_status ??
                    "Awaiting selection"}
                </p>
              </div>

              {/* ENVIRONMENT */}
              <div>
                <p className="text-muted-foreground">
                  Environmental Data
                </p>

                <p className="mt-1">
                  {intelligence?.environmental_status ??
                    "Awaiting selection"}
                </p>
              </div>

              {/* ANOMALY */}
              <div>
                <p className="text-muted-foreground">
                  Anomaly
                </p>

                <p className="mt-1">
                  {intelligence?.anomaly_status ??
                    "Awaiting selection"}
                </p>
              </div>

              {/* RISK */}
              <div>
                <p className="text-muted-foreground">
                  Flood Risk
                </p>

                <p className="mt-1">
                  {intelligence
                    ? `${intelligence.risk_level} · ${(
                        intelligence.flood_risk * 100
                      ).toFixed(1)}`
                    : "Awaiting selection"}
                </p>
              </div>

              {/* SHIFT */}
              <div>
                <p className="text-muted-foreground">
                  Distribution Shift
                </p>

                <p className="mt-1">
                  {intelligence
                    ? `${intelligence.shift_status} · ${intelligence.shift_ratio.toFixed(
                        2
                      )}×`
                    : "Awaiting selection"}
                </p>
              </div>

              {/* CONFIDENCE */}
              <div>
                <p className="text-muted-foreground">
                  Model Confidence
                </p>

                <p className="mt-1">
                  {intelligence
                    ? `${intelligence.confidence_percent.toFixed(
                        0
                      )}%`
                    : "Awaiting selection"}
                </p>
              </div>

              {/* FINAL STATUS */}
              <div>
                <p className="text-muted-foreground">
                  EarthShield Assessment
                </p>

                <p className="mt-1 font-semibold">
                  {intelligence?.earthshield_status ??
                    "Awaiting selection"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}