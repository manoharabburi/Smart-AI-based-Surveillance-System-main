import { useEffect, useRef, useState, useCallback } from 'react';

export function useAlerts(url = '/ws/alerts') {
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    const connect = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      // For development, connect to backend server on port 8000
      const host = window.location.hostname + ':8000';
      const full = url.startsWith('ws') ? url : `${protocol}://${host}${url}`;
      const ws = new WebSocket(full);
      wsRef.current = ws;
      ws.onopen = () => !cancelled && setConnected(true);
      ws.onclose = () => {
        if (!cancelled) {
          setConnected(false);
          setTimeout(connect, 2000);
        }
      };
      ws.onerror = () => ws.close();
      ws.onmessage = ev => {
        try {
          const data = JSON.parse(ev.data);
            data._receivedAt = Date.now();
            setAlerts(prev => {
              // replace if id exists
              const idx = prev.findIndex(a => a.id === data.id);
              if (idx !== -1) {
                const next = [...prev];
                // preserve original _receivedAt for ordering if earlier
                if (next[idx]._receivedAt && next[idx]._receivedAt < data._receivedAt) {
                  // keep earliest timestamp for flash logic? we want new arrival flash so keep new
                }
                next[idx] = { ...next[idx], ...data };
                return next;
              }
              const next = [...prev, data];
              return next.slice(-300);
            });
        } catch (e) {
          // ignore
        }
      };
    };
    connect();
    return () => {
      cancelled = true;
      wsRef.current && wsRef.current.close();
    };
  }, [url]);

  const resolveAlert = useCallback(async (id) => {
    try {
      const res = await fetch(`/api/alerts/${id}/resolve`, { method: 'POST' });
      if (!res.ok) throw new Error('resolve failed');
      const data = await res.json();
      setAlerts(prev => prev.map(a => a.id === id ? { ...a, ...data, resolved: true } : a));
    } catch (e) {
      // optional: toast
    }
  }, []);

  return { alerts, connected, resolveAlert };
}
