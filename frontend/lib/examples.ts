/** Example datasets shipped with the app (public/examples). */
export interface ExampleDataset {
  id: string;
  file: string;
  title: string;
  description: string;
  tags: string[];
  dateColumn: string;
  targetColumn: string;
}

export const EXAMPLES: ExampleDataset[] = [
  {
    id: 'temperatures',
    file: '/examples/daily-min-temperatures.csv',
    title: 'Daily minimum temperatures',
    description: 'Melbourne, 1981–1990. A real series with a strong yearly cycle.',
    tags: ['Daily', '3,650 rows', 'Yearly cycle'],
    dateColumn: 'Date',
    targetColumn: 'Daily minimum temperatures',
  },
  {
    id: 'sales',
    file: '/examples/sales-with-sensor.csv',
    title: 'Sales with a sensor',
    description: 'Two years of synthetic sales with a trend, and an exogenous sensor variable.',
    tags: ['Daily', '730 rows', 'Weekly cycle', 'Exogenous'],
    dateColumn: 'date',
    targetColumn: 'sales',
  },
  {
    id: 'minute-load',
    file: '/examples/minute-load.csv',
    title: 'Minute load',
    description: 'Three days of minute-level data with an hourly cycle.',
    tags: ['Minute', '4,320 rows', 'Hourly cycle'],
    dateColumn: 'timestamp',
    targetColumn: 'load',
  },
];
