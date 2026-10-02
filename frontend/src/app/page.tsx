"use client";

import { useEffect, useState } from "react";

import { Map, Satellite, Activity, Brain } from "lucide-react";

const navigation = [
  { label: "Overview", icon: Activity },
  { label: "Satellite Intelligence", icon: Satellite },
  { label: "Risk Forecast", icon: Map },
  { label: "Distribution Shift", icon: Activity },
  { label: "AI Analysis", icon: Brain },
];
export default function Home() {
  const [status, setStatus] = useState("Checking...");

useEffect(() => {
  fetch("http://127.0.0.1:8000/api/health")
    .then((res) => res.json())
    .then((data) => setStatus(data.status))
    .catch(() => setStatus("offline"));
}, []);
  return (
    <main className="flex min-h-screen bg-background text-foreground">
      <aside className="w-64 border-r p-6">
        <h1 className="text-xl font-semibold">EarthShield AI</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Multimodal Earth Intelligence
        </p>

        <nav className="mt-8 space-y-2">
          {navigation.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className="flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm hover:bg-muted"
            >
              <Icon size={18} />
              {label}
            </button>
          ))}
        </nav>
      </aside>

<section className="flex-1 p-8">
  <div>
    <h2 className="text-2xl font-semibold">Overview</h2>
    <p className="mt-2 text-muted-foreground">
      Earth observation intelligence and disaster-risk monitoring.
    </p>
  </div>

  <div className="mt-8 grid gap-4 md:grid-cols-3">
    {[
      ["Active Anomalies", "—"],
      ["Risk Status", "—"],
      ["Backend Status", status],
    ].map(([label, value]) => (
      <div key={label} className="rounded-lg border p-5">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="mt-2 text-2xl font-semibold">{value}</p>
      </div>
    ))}
  </div>

  <div className="mt-6 rounded-lg border p-6">
    <p className="text-sm text-muted-foreground">Earth Observation Map</p>
    <div className="mt-4 flex h-96 items-center justify-center rounded-md bg-muted">
      Map will be integrated here
    </div>
  </div>
</section>
    </main>
  );
}