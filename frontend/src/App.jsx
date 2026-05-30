import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useAlerts } from './hooks/useAlerts';
import { Card, Badge, Separator } from './components/ui/primitives.jsx';
import { Users, AlertTriangle, Shield, Video, Activity } from 'lucide-react';
import VideoFeed from './components/VideoFeed.jsx';
import AlertsPanel from './components/AlertsPanel.jsx';
import CameraSelector from './components/CameraSelector.jsx';

export default function App(){
  const { alerts, connected, resolveAlert } = useAlerts('/ws/alerts');
  const latest = alerts.slice(-1)[0];

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-white/10 bg-neutral-950/70 backdrop-blur sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center gap-4">
          <div className="flex items-center gap-2 text-indigo-400 font-semibold tracking-wide">
            <Shield size={22} />
            <span>Smart Surveillance System</span>
          </div>
          <div className="ml-auto flex items-center gap-4 text-xs text-neutral-400">
            <span className={connected? 'text-emerald-400':'text-red-400'}>{connected? 'ALERT STREAM LIVE':'ALERT STREAM DISCONNECTED'}</span>
            <span className="hidden sm:inline">Total Alerts: {alerts.length}</span>
          </div>
        </div>
      </header>
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 md:px-6 py-6 grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 flex flex-col gap-6">
          <Card className="relative overflow-hidden p-0">
            <div className="absolute inset-0 bg-gradient-to-br from-indigo-500/5 via-transparent to-fuchsia-500/5 pointer-events-none" />
            <div className="flex items-center justify-between px-4 pt-3 pb-2">
              <h2 className="text-sm font-semibold tracking-wide text-neutral-300 flex items-center gap-2"><Video size={16}/> LIVE FEED</h2>
              <Badge variant={connected? 'success':'danger'}>{connected? 'ONLINE':'OFFLINE'}</Badge>
            </div>
            <Separator className="my-0" />
            <VideoFeed alerts={alerts} />
          </Card>
          <StatsBar alerts={alerts} />
        </div>
        <div className="flex flex-col gap-6">
          <AlertsPanel alerts={alerts} resolveAlert={resolveAlert} />
          <CameraSelector />
          <RecentAlertHighlight alert={latest} />
        </div>
      </main>
      <footer className="text-center text-xs text-neutral-500 py-4 border-t border-white/5">
        Built with YOLOv8 + Deep SORT + FastAPI + React
      </footer>
    </div>
  );
}

function RecentAlertHighlight({alert}){
  if(!alert) return null;
  return (
    <Card className="relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-red-600/10 via-transparent to-indigo-600/10 animate-pulse" />
      <h3 className="text-xs uppercase tracking-wide mb-2 text-neutral-400 flex items-center gap-2"><Activity size={14}/> Latest Alert</h3>
      <div className="flex items-center gap-3">
        <Badge variant={badgeVariantFor(alert.severity)}>{alert.type}</Badge>
        <span className="text-xs text-neutral-400">{new Date(alert.timestamp).toLocaleTimeString()}</span>
      </div>
      <p className="mt-2 text-sm text-neutral-200 leading-relaxed">{alert.description}</p>
    </Card>
  );
}

function StatsBar({alerts}){
  const crowd = alerts.filter(a=>a.type.includes('Crowd')).length;
  const bags = alerts.filter(a=>a.type.includes('Bag')).length;
  const loiter = alerts.filter(a=>a.type.includes('Loiter')).length;
  return (
    <div className="grid grid-cols-3 gap-4">
      <MetricCard icon={<Users className="text-indigo-400" size={18}/>} label="Crowd Alerts" value={crowd} />
      <MetricCard icon={<AlertTriangle className="text-amber-400" size={18}/>} label="Abandoned Bags" value={bags} />
      <MetricCard icon={<Activity className="text-rose-400" size={18}/>} label="Loitering" value={loiter} />
    </div>
  );
}

function MetricCard({icon,label,value}){
  return (
    <Card className="p-4 flex flex-col gap-1">
      <div className="flex items-center gap-2 text-xs text-neutral-400 uppercase tracking-wide">{icon}{label}</div>
      <div className="text-2xl font-semibold text-neutral-100">{value}</div>
    </Card>
  );
}

function badgeVariantFor(sev){
  switch(sev){
    case 'high': return 'danger';
    case 'medium': return 'warning';
    case 'low': return 'info';
    default: return 'default';
  }
}
