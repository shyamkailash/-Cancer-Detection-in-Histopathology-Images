export type Model = {
  id: string;
  display_name: string;
  paradigm?: string;
  benchmark_accuracy?: number;
  checkpoint_exists?: boolean;
  architecture?: string;
  checkpoint_path?: string;
};

export type UserProfile = {
  id: number;
  username: string;
  email: string;
  is_staff: boolean;
  date_joined: string;
};

export type PredictionRecordItem = {
  id: number;
  model_name: string;
  model_display_name: string;
  predicted_class_id: number;
  predicted_class_name: string;
  confidence: number;
  normal_probability: number;
  metastasis_probability: number;
  inference_time_ms: number;
  has_explanation: boolean;
  target_layer: string;
  image_filename: string;
  overlay_preview?: string | null;
  created_at: string;
};

export type PredictionResponse = {
  status: 'success' | 'error';
  prediction: {
    class_id: number;
    class_name: string;
    confidence: number;
    probabilities: {
      normal: number;
      metastasis: number;
    };
    description: string;
  };
  model: {
    id: string;
    display_name: string;
    paradigm: string;
    device: string;
    input_size: number[];
  };
  inference_time_ms: number;
  explainability?: {
    method: string;
    target_layer: string;
    target_class: number;
    target_class_name: string;
    heatmap_base64: string;
    overlay_base64: string;
    original_base64: string;
  };
  record_id?: number;
  error?: string;
};

export type ApiResult<T> = {
  data: T | null;
  error: string | null;
  live: boolean;
  status?: number;
};

const TOKEN_KEY = 'oncolens_auth_token';
const USER_KEY = 'oncolens_auth_user';

export const authStorage = {
  getToken: (): string | null => {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  setToken: (token: string | null) => {
    try {
      if (token) localStorage.setItem(TOKEN_KEY, token);
      else localStorage.removeItem(TOKEN_KEY);
    } catch {}
  },
  getUser: (): UserProfile | null => {
    try {
      const raw = localStorage.getItem(USER_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  },
  setUser: (user: UserProfile | null) => {
    try {
      if (user) localStorage.setItem(USER_KEY, JSON.stringify(user));
      else localStorage.removeItem(USER_KEY);
    } catch {}
  },
  clear: () => {
    try {
      localStorage.removeItem(TOKEN_KEY);
      localStorage.removeItem(USER_KEY);
    } catch {}
  },
};

// Determine whether we are in a local environment
const isLocalhost =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === '0.0.0.0');

// In production, default to the official Render API if VITE_API_BASE_URL is not set
const defaultApiUrl = isLocalhost ? '' : 'https://cancer-detection-api.onrender.com';
const base = (import.meta.env.VITE_API_BASE_URL || defaultApiUrl).replace(/\/$/, '');

async function request<T>(
  path: string,
  init?: RequestInit,
  timeoutMs = 60000
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  let isTimedOut = false;
  const timer = window.setTimeout(() => {
    isTimedOut = true;
    controller.abort();
  }, timeoutMs);

  const token = authStorage.getToken();
  const headers = new Headers(init?.headers);
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Token ${token}`);
  }

  try {
    const res = await fetch(`${base}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });

    const contentType = res.headers.get('content-type') || '';
    let data: any = null;

    if (contentType.includes('application/json')) {
      data = await res.json();
    } else {
      const text = await res.text();
      try {
        data = JSON.parse(text);
      } catch {
        // If the server returned HTML (e.g. Cloudflare error or Vercel SPA routing fallback)
        if (text.trim().startsWith('<')) {
          const titleMatch = text.match(/<title>([^<]+)<\/title>/i);
          const title = titleMatch ? titleMatch[1].trim() : `Server returned HTML (${res.status})`;
          return {
            data: null,
            error:
              res.status >= 500
                ? `Backend service error (${res.status}): ${title}. The service may be waking up from cold sleep.`
                : `Unexpected HTML response from ${base || 'server'} (${res.status}): ${title}`,
            live: false,
            status: res.status,
          };
        }
        data = text ? { message: text } : null;
      }
    }

    if (!res.ok) {
      // Extract human-readable error from DRF payload
      let errorMsg = `Request failed (HTTP ${res.status})`;
      if (data && typeof data === 'object') {
        if (typeof data.error === 'string') errorMsg = data.error;
        else if (typeof data.detail === 'string') errorMsg = data.detail;
        else if (typeof data.message === 'string') errorMsg = data.message;
        else if (Array.isArray(data.non_field_errors)) errorMsg = data.non_field_errors.join(' ');
        else {
          const firstKey = Object.keys(data)[0];
          if (firstKey) {
            const val = data[firstKey];
            errorMsg = `${firstKey}: ${Array.isArray(val) ? val.join(' ') : val}`;
          }
        }
      }
      return { data: null, error: errorMsg, live: false, status: res.status };
    }

    return { data, error: null, live: true, status: res.status };
  } catch (error: any) {
    let msg = 'Backend unavailable';
    if (isTimedOut || error.name === 'AbortError' || controller.signal.aborted) {
      msg =
        'Request timed out. The Render backend may be waking up from cold sleep (free tier spins down on inactivity). Please wait a few seconds and try again.';
    } else if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
      msg = 'Network connection to backend failed. Please verify internet connection or CORS settings.';
    } else if (error instanceof Error) {
      msg = error.message;
    }
    return { data: null, error: msg, live: false };
  } finally {
    window.clearTimeout(timer);
  }
}

export const api = {
  getBaseUrl: () => base,

  health: () =>
    request<{ status: string; version: string; environment: string }>(
      '/api/health/',
      undefined,
      30000
    ),

  models: () =>
    request<{ status: string; count: number; models: Model[] }>(
      '/api/models/',
      undefined,
      30000
    ),

  predict: (file: File, model: string, includeGradcam = true) => {
    const form = new FormData();
    form.append('image', file);
    form.append('model_name', model);
    form.append('include_gradcam', includeGradcam ? 'true' : 'false');
    // Notice: Do NOT pass manual Content-Type header so browser sets multipart boundary automatically
    return request<PredictionResponse>(
      '/api/predict/',
      {
        method: 'POST',
        body: form,
      },
      90000 // 90-second timeout to accommodate PyTorch cold starts on low-memory containers
    );
  },

  predictions: {
    history: (limit = 50) =>
      request<{ status: string; count: number; results: PredictionRecordItem[] }>(
        `/api/predict/history/?limit=${limit}`,
        undefined,
        30000
      ),
    detail: (id: number) =>
      request<PredictionRecordItem>(`/api/predict/history/${id}/`, undefined, 30000),
    delete: (id: number) =>
      request<{ status: string }>(`/api/predict/history/${id}/`, { method: 'DELETE' }, 30000),
    clearAll: () =>
      request<{ status: string; message: string; deleted_count: number }>(
        '/api/predict/history/',
        { method: 'DELETE' },
        30000
      ),
  },

  auth: {
    login: (username: string, password: string) =>
      request<{ token: string; user: UserProfile }>(
        '/api/auth/login/',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        },
        30000
      ),
    register: (username: string, email: string, password: string) =>
      request<{ token: string; user: UserProfile }>(
        '/api/auth/register/',
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password }),
        },
        30000
      ),
    logout: () =>
      request<{ status: string; detail: string }>(
        '/api/auth/logout/',
        {
          method: 'POST',
        },
        30000
      ),
    me: () => request<UserProfile>('/api/auth/me/', undefined, 30000),
  },

  mlops: {
    status: () => request<Record<string, any>>('/api/mlops/status/', undefined, 30000),
    models: () => request<{ status: string; count: number; models: any[] }>('/api/mlops/models/', undefined, 30000),
    runs: () => request<{ status: string; count: number; runs: any[] }>('/api/mlops/runs/', undefined, 30000),
    drift: () => request<Record<string, any>>('/api/mlops/drift/', undefined, 30000),
  },
};
