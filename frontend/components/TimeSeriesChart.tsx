'use client';

import React from 'react';
import dynamic from 'next/dynamic';
import { Layout, Data } from 'plotly.js';

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface PredictionSeries {
  name: string;
  data: any[];
  color?: string;
}

interface TimeSeriesChartProps {
  data: any[];
  dateColumn: string;
  targetColumn: string;
  title?: string;
  predictions?: PredictionSeries[];
}

export default function TimeSeriesChart({ data, dateColumn, targetColumn, title, predictions }: TimeSeriesChartProps) {
  // Data is already normalized with ISO format dates from backend
  // Just extract the date strings directly
  const xValues = data.map(row => {
    const dateVal = row[dateColumn];
    // Handle both string and Date objects, extract YYYY-MM-DD
    if (!dateVal) return '';
    if (typeof dateVal === 'string') return dateVal.split('T')[0];
    if (dateVal instanceof Date) return dateVal.toISOString().split('T')[0];
    return String(dateVal);
  });
  const yValues = data.map(row => row[targetColumn]);

  const plotData: Data[] = [
    {
      x: xValues,
      y: yValues,
      type: 'scatter',
      mode: 'lines',
      marker: { color: '#f59e0b' }, // Amber-500
      line: { width: 2, shape: 'spline' },
      name: 'Actual',
    }
  ];

  if (predictions && predictions.length > 0) {
    predictions.forEach((pred, idx) => {
      // Predictions also have ISO format dates from backend
      const predX = pred.data.map(row => {
        const dateVal = row[dateColumn];
        if (!dateVal) return '';
        if (typeof dateVal === 'string') return dateVal.split('T')[0];
        return String(dateVal);
      });
      const predY = pred.data.map(row => row['prediction'] !== undefined ? row['prediction'] : row[targetColumn]);
      
      // Generate a color if not provided
      const colors = ['#3b82f6', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4'];
      const color = pred.color || colors[idx % colors.length];

      plotData.push({
        x: predX,
        y: predY,
        type: 'scatter',
        mode: 'lines',
        marker: { color: color },
        line: { width: 2, dash: 'dot' },
        name: pred.name,
      });
    });
  }

  const layout: Partial<Layout> = {
    title: {
      text: title || 'Time Series Preview',
      font: { color: '#e2e8f0' }
    },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, sans-serif',
      color: '#94a3b8' // Slate-400
    },
    xaxis: {
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
      showgrid: true,
    },
    yaxis: {
      gridcolor: 'rgba(255,255,255,0.1)',
      zerolinecolor: 'rgba(255,255,255,0.1)',
      showgrid: true,
    },
    margin: { t: 40, r: 20, l: 40, b: 40 },
    autosize: true,
    hovermode: 'x unified',
  };

  return (
    <div className="w-full h-[300px] rounded-xl overflow-hidden border border-white/10 bg-black/20">
      <Plot
        data={plotData}
        layout={layout}
        useResizeHandler={true}
        style={{ width: '100%', height: '100%' }}
        config={{ displayModeBar: false }}
      />
    </div>
  );
}
