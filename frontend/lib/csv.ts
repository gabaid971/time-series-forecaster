import Papa from 'papaparse';

/**
 * Parse a CSV file and return the data.
 */
export function parseCSV(file: File): Promise<{
  data: any[];
  columns: string[];
  filename: string;
}> {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          console.warn('CSV parsing warnings:', results.errors);
        }
        
        resolve({
          data: results.data,
          columns: results.meta.fields || [],
          filename: file.name,
        });
      },
      error: (error) => {
        reject(new Error(`Failed to parse CSV: ${error.message}`));
      },
    });
  });
}

/**
 * Export data to CSV and trigger download.
 */
export function exportToCSV(data: any[], filename: string): void {
  if (data.length === 0) return;

  const headers = Object.keys(data[0]);
  const csvContent = [
    headers.join(','),
    ...data.map(row => 
      headers.map(h => {
        const val = row[h];
        if (val === null || val === undefined) return '';
        if (typeof val === 'string' && val.includes(',')) return `"${val}"`;
        return val;
      }).join(',')
    )
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

/**
 * Detect likely date and target columns from column names.
 */
export function detectColumns(columns: string[]): {
  dateColumn: string | null;
  targetColumn: string | null;
} {
  const datePatterns = ['date', 'time', 'timestamp', 'datetime', 'dt', 'ds'];
  const targetPatterns = ['value', 'target', 'y', 'price', 'sales', 'count', 'amount'];

  const dateColumn = columns.find(c => 
    datePatterns.some(p => c.toLowerCase().includes(p))
  ) || null;

  const targetColumn = columns.find(c => 
    targetPatterns.some(p => c.toLowerCase().includes(p)) && c !== dateColumn
  ) || columns.find(c => c !== dateColumn) || null;

  return { dateColumn, targetColumn };
}
