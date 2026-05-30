import React, { useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card, Badge, Separator } from './ui/primitives.jsx';

const sevToVariant = sev => {
  switch(sev){
    case 'high': return 'danger';
    case 'critical': return 'danger';
    case 'medium': return 'warning';
    case 'low': return 'info';
    default: return 'default';
  }
};

export default function AlertsPanel({alerts, resolveAlert}){
  const last100 = useMemo(()=>alerts.slice(-100).reverse(), [alerts]);
  return (
    <Card className="flex flex-col h-[480px]">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-xs font-semibold tracking-wide text-neutral-400">ALERTS FEED</h2>
        <Badge variant="info">{alerts.length}</Badge>
      </div>
      <Separator />
      <div className="relative overflow-y-auto custom-scroll -m-2 -mt-1 p-2 space-y-2 pr-3">
        <AnimatePresence initial={false}>
          {last100.map(a => <AlertRow key={a.id + '_' + a.timestamp + '_' + (a.resolved? 'r':'u')} alert={a} onResolve={resolveAlert} />)}
        </AnimatePresence>
        {last100.length === 0 && <div className="text-xs text-neutral-500 py-8 text-center">No alerts yet.</div>}
      </div>
    </Card>
  );
}

function ownerInfo(alert){
  if(!alert?.data) return null;
  if(alert.type.toLowerCase().includes('knife') && alert.data.nearest_person_id){
    return <span className="text-[10px] text-red-300/80 ml-2">person {alert.data.nearest_person_id}</span>;
  }
  if(alert.type.toLowerCase().includes('bag') && alert.data.owner_person_id){
    return <span className="text-[10px] text-amber-300/80 ml-2">owner person {alert.data.owner_person_id}</span>;
  }
  return null;
}

function AlertRow({alert,onResolve}){
  const canResolve = !alert.resolved && (
    alert.type.toLowerCase().includes('knife') ||
    alert.type.toLowerCase().includes('fighting') ||
    alert.severity === 'critical'
  );
  return (
    <motion.div
      layout
      initial={{opacity:0, y:6, scale:0.98}}
      animate={{opacity:1, y:0, scale:1}}
      exit={{opacity:0, y:-6}}
      className={`group rounded-md border border-white/10 bg-neutral-800/40 px-3 py-2 text-xs backdrop-blur relative ${Date.now()-alert._receivedAt < 1500 ? 'animate-flash':''} ${alert.resolved? 'opacity-60 line-through decoration-neutral-500/40':''}`}
    >
      <div className="flex items-center gap-2">
        <Badge variant={sevToVariant(alert.severity)}>{alert.type}</Badge>
        {alert.resolved && <Badge variant="default" className="bg-emerald-600/40 text-emerald-200">resolved</Badge>}
        <span className="font-mono text-[10px] text-neutral-400">{new Date(alert.timestamp).toLocaleTimeString()}</span>
        <span className="ml-auto text-[10px] text-neutral-500 uppercase tracking-wide">{alert.camera_id}</span>
      </div>
      <p className="mt-1 text-neutral-200 leading-snug flex items-center flex-wrap gap-1">
        <span>{alert.description}</span>
        {ownerInfo(alert)}
        {canResolve && (
          <button
            onClick={()=>onResolve(alert.id)}
            className="ml-auto text-[10px] px-2 py-0.5 rounded bg-red-600/70 hover:bg-red-600 text-white font-medium tracking-wide transition-colors"
          >Resolve</button>
        )}
      </p>
    </motion.div>
  );
}
