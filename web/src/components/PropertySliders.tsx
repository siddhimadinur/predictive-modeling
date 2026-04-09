"use client";

interface PropertySlidersProps {
  rooms: number;
  bedrooms: number;
  householdSize: number;
  onRoomsChange: (val: number) => void;
  onBedroomsChange: (val: number) => void;
  onHouseholdSizeChange: (val: number) => void;
}

function Slider({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (val: number) => void;
}) {
  return (
    <div>
      <div className="flex justify-between items-center mb-2">
        <label className="text-sm font-semibold text-gray-700">{label}</label>
        <span className="text-2xl font-bold text-sunset-500">{value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={1}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 rounded-lg appearance-none cursor-pointer accent-sunset-500"
      />
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

export default function PropertySliders({
  rooms,
  bedrooms,
  householdSize,
  onRoomsChange,
  onBedroomsChange,
  onHouseholdSizeChange,
}: PropertySlidersProps) {
  return (
    <div className="space-y-6">
      <Slider
        label="Total Rooms"
        value={rooms}
        min={1}
        max={5}
        onChange={(val) => {
          onRoomsChange(val);
          if (bedrooms > val) onBedroomsChange(val);
        }}
      />
      <Slider
        label="Bedrooms"
        value={bedrooms}
        min={1}
        max={rooms}
        onChange={onBedroomsChange}
      />
      <Slider
        label="People in Household"
        value={householdSize}
        min={1}
        max={8}
        onChange={onHouseholdSizeChange}
      />
    </div>
  );
}
