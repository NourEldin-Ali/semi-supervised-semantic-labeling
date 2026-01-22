import type { RunStats } from '../services/api';

interface RunStatisticsProps {
  stats: RunStats | null | undefined;
  className?: string;
}

export default function RunStatistics({ stats, className = '' }: RunStatisticsProps) {
  if (!stats) return null;
  const hasAny =
    stats.execution_time_seconds != null ||
    stats.tokens_consumed != null ||
    stats.energy_consumed_kwh != null;
  if (!hasAny) return null;

  return (
    <div className={`p-4 bg-slate-50 rounded-lg border border-slate-200 ${className}`}>
      <h3 className="text-sm font-semibold text-slate-700 mb-3">Run statistics</h3>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {stats.execution_time_seconds != null && (
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide">Execution time</p>
            <p className="text-lg font-medium text-slate-900">
              {stats.execution_time_seconds.toFixed(2)} s
            </p>
          </div>
        )}
        {stats.tokens_consumed != null && (
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide">Tokens consumed</p>
            <p className="text-lg font-medium text-slate-900">
              {stats.tokens_consumed.toLocaleString()}
            </p>
          </div>
        )}
        {stats.energy_consumed_kwh != null && (
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide">Energy consumed</p>
            <p className="text-lg font-medium text-slate-900">
              {stats.energy_consumed_kwh.toFixed(6)} kWh
            </p>
            {stats.emissions_kg_co2eq != null && stats.emissions_kg_co2eq > 0 && (
              <p className="text-xs text-slate-600 mt-0.5">
                {stats.emissions_kg_co2eq.toFixed(6)} kg CO₂eq
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
