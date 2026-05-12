/**
 * Color utilities for segmentation visualization.
 * Matches the Python CLASS_COLORS palette.
 */

export const CONSTRUCTION_CLASS_COLORS: Record<string, string> = {
  background:      '#000000',
  sky:             '#87ceeb',
  ground:          '#8b7765',
  wall:            '#c0c0c0',
  floor:           '#d3d3d3',
  soil:            '#a0522d',
  walkable_path:   '#90ee90',
  concrete:        '#a9a9a9',
  steel_rebar:     '#b8860b',
  formwork:        '#d2b48c',
  scaffolding:     '#ffa500',
  active_zone:     '#32cd32',
  restricted_zone: '#ff8c00',
  unsafe_edge:     '#ff4500',
  worker:          '#0000ff',
  machinery:       '#800080',
  crane:           '#ffd700',
  excavator:       '#008000',
  vehicle:         '#00ffff',
  hazard_zone:     '#ff0000',
  open_shaft:      '#9400d3',
};

export const RISK_LEVEL_COLORS: Record<string, string> = {
  low:      '#10b981',
  medium:   '#f59e0b',
  high:     '#f97316',
  critical: '#ef4444',
};

export function riskToHex(score: number): string {
  if (score >= 0.75) return RISK_LEVEL_COLORS.critical;
  if (score >= 0.50) return RISK_LEVEL_COLORS.high;
  if (score >= 0.25) return RISK_LEVEL_COLORS.medium;
  return RISK_LEVEL_COLORS.low;
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}
