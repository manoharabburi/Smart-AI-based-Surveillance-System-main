import React, { useMemo } from 'react';

export default function VideoFeed({ alerts = [] }){
  const recent = useMemo(()=>{
    const now = Date.now();
    return alerts.filter(a=> now - a._receivedAt < 15000); // last 15s
  },[alerts]);
  const knifeAlert = recent.find(a=> a.type && a.type.toLowerCase().includes('knife') && !a.resolved);
  const resolvedKnifeAlert = !knifeAlert && recent.find(a=> a.type && a.type.toLowerCase().includes('knife') && a.resolved);
  const stationaryBagAlert = recent.find(a=> a.type && a.type.toLowerCase().includes('stationary bag'));

  return (
    <div className="relative w-full h-[560px] md:h-[640px] overflow-hidden rounded-b-lg border-t border-white/5 bg-black">
      <img
        src="/video_feed"
        alt="Live Feed"
        className="w-full h-full object-contain select-none"
        draggable={false}
        style={{background:'radial-gradient(circle at center, #111 0%, #000 60%)'}}
      />
      <div className="absolute top-2 right-2 text-[11px] font-mono bg-red-600/80 text-white px-2 py-0.5 rounded animate-pulse shadow">LIVE</div>
      {knifeAlert && (
        <div className="absolute inset-x-0 top-0 bg-gradient-to-r from-red-800/90 via-red-600/90 to-red-800/90 text-center py-3 animate-pulse shadow-xl z-20">
          <div className="text-sm font-semibold tracking-wide text-red-100">CRITICAL: KNIFE DETECTED</div>
          <div className="text-[11px] text-red-200/80">Immediate attention required {knifeAlert.data?.nearest_person_id && <span className="ml-1">(near person {knifeAlert.data.nearest_person_id})</span>}</div>
        </div>
      )}
      {!knifeAlert && resolvedKnifeAlert && (
        <div className="absolute inset-x-0 top-0 bg-green-700/80 backdrop-blur-sm text-center py-2 z-20 shadow">
          <span className="text-xs font-semibold text-emerald-100 tracking-wide">Knife alert resolved</span>
        </div>
      )}
      {!knifeAlert && !resolvedKnifeAlert && stationaryBagAlert && (
        <div className="absolute top-0 left-0 right-0 bg-amber-600/80 text-center py-2 z-10 backdrop-blur-sm">
          <span className="text-xs font-medium text-amber-100">Stationary Bag Alert {stationaryBagAlert.data?.owner_person_id && <span className="ml-1">(owner person {stationaryBagAlert.data.owner_person_id})</span>}</span>
        </div>
      )}
      <div className="pointer-events-none absolute inset-0">
        {/* subtle vignette */}
        <div className="absolute inset-0 bg-gradient-to-b from-black/10 via-transparent to-black/40" />
      </div>
    </div>
  );
}
