"use client";

import {
  CircleMarker,
  MapContainer,
  TileLayer,
  useMapEvents,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

type Props = {
  position: [number, number] | null;
  onSelect: (position: [number, number]) => void;
};

function MapClick({ onSelect }: { onSelect: Props["onSelect"] }) {
  useMapEvents({
    click: (event) => {
      onSelect([event.latlng.lat, event.latlng.lng]);
    },
  });

  return null;
}

export default function EarthMap({ position, onSelect }: Props) {
  return (
    <MapContainer
      center={[20.5937, 78.9629]}
      zoom={5}
      className="h-full w-full"
    >
      <TileLayer
        attribution="&copy; OpenStreetMap contributors"
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      <MapClick onSelect={onSelect} />

      {position && (
        <CircleMarker center={position} radius={8} pathOptions={{ color: "red" }} />
      )}
    </MapContainer>
  );
}