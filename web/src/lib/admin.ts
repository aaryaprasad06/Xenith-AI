const KEY = "face-id-admin-token";

export function readToken(): string {
  return sessionStorage.getItem(KEY) ?? "";
}

export function saveToken(token: string): void {
  sessionStorage.setItem(KEY, token.trim());
}

export function authHeaders(): HeadersInit {
  const token = readToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
}
