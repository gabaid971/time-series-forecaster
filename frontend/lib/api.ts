/**
 * API Configuration for the Time Series Forecaster frontend.
 *
 * The browser calls the backend directly. The backend is public (demo app):
 * there is no secret here, abuse is limited server-side (CORS, request size limits).
 */

export const BACKEND_URL = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');

export const getApiUrl = (endpoint: string): string => `${BACKEND_URL}/${endpoint}`;

export const API_HEADERS: HeadersInit = { 'Content-Type': 'application/json' };

/**
 * Extract a readable error message from a failed backend response.
 * FastAPI errors look like {"detail": "..."} or {"detail": [{"msg": "..."}]}.
 */
export async function getErrorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg: string }) => d.msg).join('; ');
  } catch {
    // Body is not JSON
  }
  return `Backend error ${response.status}`;
}
