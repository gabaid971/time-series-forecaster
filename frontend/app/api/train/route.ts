import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    
    // On récupère les variables CACHÉES (sans NEXT_PUBLIC)
    const BACKEND_URL = process.env.BACKEND_PRIVATE_URL; 
    const API_KEY = process.env.BACKEND_API_KEY;

    const response = await fetch(`${BACKEND_URL}/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': API_KEY || '',
      },
      body: JSON.stringify(body),
    });

    const data = await response.json();
    return NextResponse.json(data, { status: response.status });
    
  } catch (error) {
    return NextResponse.json({ error: "Erreur serveur proxy" }, { status: 500 });
  }
}