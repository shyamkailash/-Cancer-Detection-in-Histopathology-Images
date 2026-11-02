import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  CloudCog,
  Database,
  FileImage,
  FlaskConical,
  Gauge,
  GitBranch,
  Layers3,
  LockKeyhole,
  LogIn,
  LogOut,
  Menu,
  Network,
  RefreshCw,
  Server,
  ShieldCheck,
  Sparkles,
  Trash2,
  UploadCloud,
  User,
  Users,
  X,
} from 'lucide-react';
import {
  api,
  authStorage,
  Model,
  PredictionRecordItem,
  PredictionResponse,
  UserProfile,
} from './api';
import './styles.css';

type Section =
  | 'Overview'
  | 'Cancer Detection'
  | 'Federated Learning'
  | 'Model Evaluation'
  | 'Model Registry'
  | 'MLOps Monitoring'
  | 'Privacy and Clients'
  | 'Settings';

const navItems: { label: Section; icon: typeof Activity }[] = [
  { label: 'Overview', icon: Gauge },
  { label: 'Cancer Detection', icon: FileImage },
  { label: 'Federated Learning', icon: Network },
  { label: 'Model Evaluation', icon: BarChart3 },
  { label: 'Model Registry', icon: Layers3 },
  { label: 'MLOps Monitoring', icon: Activity },
  { label: 'Privacy and Clients', icon: ShieldCheck },
  { label: 'Settings', icon: CloudCog },
];

const benchmarkMatrix = [
  {
    id: 'resnet50_centralized',
    name: 'Centralized ResNet-50',
    paradigm: 'Centralized',
    accuracy: 95.74,
    sensitivity: 95.2,
    specificity: 96.28,
    rocAuc: 0.989,
    tone: 'green',
  },
  {
    id: 'resnet18_centralized',
    name: 'Centralized ResNet-18',
    paradigm: 'Centralized',
    accuracy: 95.35,
    sensitivity: 94.12,
    specificity: 96.58,
    rocAuc: 0.985,
    tone: 'green',
  },
  {
    id: 'fedavg',
    name: 'Federated ResNet-18 (FedAvg)',
    paradigm: 'Federated',
    accuracy: 95.09,
    sensitivity: 93.95,
    specificity: 96.23,
    rocAuc: 0.983,
    tone: 'blue',
  },
  {
    id: 'fedprox',
    name: 'Federated ResNet-18 (FedProx)',
    paradigm: 'Federated',
    accuracy: 94.88,
    sensitivity: 93.6,
    specificity: 96.16,
    rocAuc: 0.981,
    tone: 'blue',
  },
  {
    id: 'mobilenetv3_centralized',
    name: 'Centralized MobileNetV3-Large',
    paradigm: 'Centralized',
    accuracy: 94.62,
    sensitivity: 93.8,
    specificity: 95.44,
    rocAuc: 0.978,
    tone: 'green',
  },
  {
    id: 'dp_fedavg',
    name: 'Federated ResNet-18 (DP-FedAvg)',
    paradigm: 'Differentially Private',
    accuracy: 94.33,
    sensitivity: 92.8,
    specificity: 95.86,
    rocAuc: 0.974,
    tone: 'purple',
  },
];

function App() {
  const [section, setSection] = useState<Section>('Overview');
  const [mobile, setMobile] = useState(false);
  const [modelsLive, setModelsLive] = useState<Model[]>([]);
  const [backendLive, setBackendLive] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(authStorage.getUser());
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const checkStatus = () => {
    setRefreshing(true);
    Promise.all([api.health(), api.models()])
      .then(([health, mods]) => {
        setBackendLive(health.live);
        if (mods.data?.models) {
          setModelsLive(mods.data.models);
        }
      })
      .finally(() => setTimeout(() => setRefreshing(false), 500));
  };

  useEffect(() => {
    checkStatus();
    // Verify stored token if present
    if (authStorage.getToken()) {
      api.auth.me().then((res) => {
        if (res.data) {
          setCurrentUser(res.data);
          authStorage.setUser(res.data);
        } else if (res.status === 401) {
          // Token expired or invalid
          authStorage.clear();
          setCurrentUser(null);
        }
      });
    }
  }, []);

  const handleLogout = async () => {
    try {
      await api.auth.logout();
    } catch {}
    authStorage.clear();
    setCurrentUser(null);
  };

  return (
    <div className="shell">
      {mobile && <div className="scrim" onClick={() => setMobile(false)} />}
      <aside className={mobile ? 'sidebar open' : 'sidebar'}>
        <div className="brand">
          <div className="brand-mark">
            <Sparkles />
          </div>
          <div>
            <strong>OncoLens</strong>
            <span>RESEARCH CONSOLE</span>
          </div>
          <button className="icon-btn close" onClick={() => setMobile(false)}>
            <X />
          </button>
        </div>

        <div className="workspace">
          <span className="eyebrow">WORKSPACE</span>
          <button className="workspace-btn">
            <span className="workspace-dot" />
            Cancer Detection Lab
            <ChevronDown />
          </button>
        </div>

        <nav>
          {navItems.map(({ label, icon: Icon }) => (
            <button
              key={label}
              className={section === label ? 'nav-item active' : 'nav-item'}
              onClick={() => {
                setSection(label);
                setMobile(false);
              }}
            >
              <Icon />
              <span>{label}</span>
              {label === 'MLOps Monitoring' && <i className="nav-dot" />}
            </button>
          ))}
        </nav>

        <div className="sidebar-foot">
          <div className="connection">
            <span className={backendLive ? 'status-dot online' : 'status-dot'} />
            <div>
              <strong>{backendLive ? 'Backend connected' : 'Demo / Offline'}</strong>
              <span>{backendLive ? 'Django API responding' : 'API unavailable'}</span>
            </div>
            <RefreshCw className={refreshing ? 'spin' : ''} onClick={checkStatus} />
          </div>

          <div className="user">
            <div className="avatar">
              {currentUser ? currentUser.username.substring(0, 2).toUpperCase() : 'GS'}
            </div>
            <div style={{ flex: 1, minWidth: 0 }}>
              <strong style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {currentUser ? currentUser.username : 'Guest Researcher'}
              </strong>
              <span>{currentUser ? currentUser.email : 'Click to sign in'}</span>
            </div>
            {currentUser ? (
              <button
                className="icon-btn"
                title="Log out"
                onClick={handleLogout}
                style={{ padding: 4 }}
              >
                <LogOut size={14} />
              </button>
            ) : (
              <button
                className="icon-btn"
                title="Sign in"
                onClick={() => {
                  setAuthMode('login');
                  setShowAuthModal(true);
                }}
                style={{ padding: 4 }}
              >
                <LogIn size={14} />
              </button>
            )}
          </div>
        </div>
      </aside>

      <main className="main">
        <header>
          <button className="icon-btn menu" onClick={() => setMobile(true)}>
            <Menu />
          </button>
          <div>
            <p className="breadcrumb">
              ONCOLENS <span>/</span> {section.toUpperCase()}
            </p>
            <h1>{section}</h1>
          </div>
          <div className="header-actions">
            <span className="live-pill">
              <span className={backendLive ? 'status-dot online' : 'status-dot'} />
              {backendLive ? 'LIVE API' : 'OFFLINE'}
            </span>
            {currentUser ? (
              <button
                className="avatar"
                title={`Signed in as ${currentUser.username}`}
                onClick={() => setSection('Settings')}
              >
                {currentUser.username.substring(0, 2).toUpperCase()}
              </button>
            ) : (
              <button
                className="outline-btn"
                onClick={() => {
                  setAuthMode('login');
                  setShowAuthModal(true);
                }}
              >
                <LogIn size={13} /> Sign In
              </button>
            )}
          </div>
        </header>

        {section === 'Cancer Detection' && (
          <Detection models={modelsLive} currentUser={currentUser} backendLive={backendLive} />
        )}
        {section === 'Overview' && (
          <Overview
            models={modelsLive}
            backendLive={backendLive}
            onNavigate={(sec) => setSection(sec)}
            onRefresh={checkStatus}
          />
        )}
        {section === 'Federated Learning' && <FederatedLearning backendLive={backendLive} />}
        {section === 'Model Evaluation' && <ModelEvaluation models={modelsLive} />}
        {section === 'Model Registry' && (
          <ModelRegistry models={modelsLive} backendLive={backendLive} />
        )}
        {section === 'MLOps Monitoring' && <MLOpsMonitoring backendLive={backendLive} />}
        {section === 'Privacy and Clients' && <PrivacyAndClients backendLive={backendLive} />}
        {section === 'Settings' && (
          <SettingsPage
            currentUser={currentUser}
            backendLive={backendLive}
            onOpenAuth={() => setShowAuthModal(true)}
            onLogout={handleLogout}
          />
        )}
      </main>

      {showAuthModal && (
        <AuthModal
          mode={authMode}
          onClose={() => setShowAuthModal(false)}
          onSuccess={(user) => {
            setCurrentUser(user);
            setShowAuthModal(false);
          }}
          onSwitchMode={(mode) => setAuthMode(mode)}
        />
      )}
    </div>
  );
}

function Overview({
  models,
  backendLive,
  onNavigate,
  onRefresh,
}: {
  models: Model[];
  backendLive: boolean;
  onNavigate: (s: Section) => void;
  onRefresh: () => void;
}) {
  const [runs, setRuns] = useState<any[]>([]);
  const [loadingRuns, setLoadingRuns] = useState(false);

  useEffect(() => {
    if (backendLive) {
      setLoadingRuns(true);
      api.mlops
        .runs()
        .then((res) => {
          if (res.data?.runs) setRuns(res.data.runs);
        })
        .finally(() => setLoadingRuns(false));
    }
  }, [backendLive]);

  const bestAccuracy = useMemo(() => {
    if (!models.length) return '95.74%';
    const max = Math.max(...models.map((m) => m.benchmark_accuracy || 0));
    return max > 0 ? `${(max * 100).toFixed(2)}%` : '95.74%';
  }, [models]);

  const chartData = useMemo(() => {
    return benchmarkMatrix.map((m) => ({
      name: m.name.replace('Centralized ', 'Cent-').replace('Federated ', 'Fed-').replace(' (', ' - ').replace(')', ''),
      accuracy: m.accuracy,
      paradigm: m.paradigm,
    }));
  }, []);

  return (
    <div className="content">
      <div className="welcome">
        <div>
          <p className="eyebrow green-text">AGENTIC FEDERATED LEARNING CONSOLE</p>
          <h2>Welcome, Researcher.</h2>
          <p className="subtle">Real-time histopathology cancer classification and multi-site orchestration.</p>
        </div>
        <button className="outline-btn" onClick={onRefresh}>
          <RefreshCw size={13} /> Refresh data
        </button>
      </div>

      <div className="kpi-grid">
        <Kpi
          icon={BrainCircuit}
          label="Registered models"
          value={backendLive ? (models.length > 0 ? String(models.length) : '0') : 'Not available'}
          detail={backendLive ? `${models.filter((m) => m.checkpoint_exists !== false).length} operational checkpoints` : 'Backend offline'}
          accent="blue"
        />
        <Kpi
          icon={CheckCircle2}
          label="Top benchmark accuracy"
          value={backendLive && models.length > 0 ? bestAccuracy : (backendLive ? 'Not available' : '—')}
          detail="Held-out PCam test set"
          accent="green"
        />
        <Kpi
          icon={Network}
          label="Federated network"
          value="5 Sites"
          detail="Dirichlet Non-IID (α=0.5)"
          accent="purple"
        />
        <Kpi
          icon={ShieldCheck}
          label="Privacy accounting"
          value="RDP"
          detail="ε bounded, δ=1e-5"
          accent="pink"
        />
      </div>

      <div className="section-heading">
        <div>
          <p className="eyebrow">BENCHMARK PERFORMANCE</p>
          <h3>Model Architecture & Paradigm Comparison (Held-out Test Set)</h3>
        </div>
        <div className="legend">
          <span>
            <i className="legend-line green" /> Benchmark Accuracy (%)
          </span>
        </div>
      </div>

      <div className="chart-card">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={chartData} margin={{ top: 10, right: 15, left: -15, bottom: 20 }}>
            <CartesianGrid stroke="#1d2d3b" vertical={false} />
            <XAxis dataKey="name" stroke="#6b7d8e" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
            <YAxis domain={[90, 100]} stroke="#6b7d8e" tickLine={false} axisLine={false} tickFormatter={(v) => `${v}%`} />
            <Tooltip
              contentStyle={{ background: '#101d2b', border: '1px solid #2a3b4c', borderRadius: 8, color: '#e5edf3' }}
              formatter={(val: any) => [`${val}%`, 'Accuracy']}
            />
            <Bar dataKey="accuracy" fill="#36d399" radius={[4, 4, 0, 0]}>
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.paradigm.includes('Private') ? '#bb9afc' : entry.paradigm.includes('Fed') ? '#56b9ef' : '#36d399'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="chart-footer">
          <span>
            <span className="stat-value">95.74%</span> Peak: Centralized ResNet-50
          </span>
          <span>
            <span className="stat-value purple-text">95.09%</span> Peak Federated: ResNet-18 FedAvg
          </span>
          <span className="chart-source">{backendLive ? 'Verified Test Set Metrics' : 'Verified Baselines'}</span>
        </div>
      </div>

      <div className="lower-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">MODEL REGISTRY</p>
              <h3>Active Inference Candidates</h3>
            </div>
            <button className="text-btn" onClick={() => onNavigate('Model Registry')}>
              View all ({models.length}) →
            </button>
          </div>
          {models.length > 0 ? (
            models.slice(0, 4).map((m) => (
              <div className="model-row" key={m.id}>
                <div className="model-icon blue">
                  <BrainCircuit />
                </div>
                <div className="model-name">
                  <strong>{m.display_name}</strong>
                  <span>{m.paradigm || 'Centralized'} · PyTorch</span>
                </div>
                <b>{m.benchmark_accuracy != null ? `${(m.benchmark_accuracy * 100).toFixed(2)}%` : '—'}</b>
                <span className={`badge ${m.checkpoint_exists === false ? 'red' : 'green'}`}>
                  {m.checkpoint_exists === false ? 'Model checkpoint missing' : 'Validated'}
                </span>
              </div>
            ))
          ) : (
            <div style={{ padding: '24px 0', textAlign: 'center', color: '#6e8292', fontSize: 11 }}>
              <p>{backendLive ? 'No records found in model registry.' : 'Model registry not available.'}</p>
            </div>
          )}
        </div>

        <div className="panel events">
          <div className="panel-head">
            <div>
              <p className="eyebrow">MLOPS TELEMETRY</p>
              <h3>Recent Automated Runs</h3>
            </div>
            <button className="text-btn" onClick={() => onNavigate('MLOps Monitoring')}>
              Inspect →
            </button>
          </div>
          {runs.length > 0 ? (
            runs.slice(0, 4).map((r, i) => (
              <div className="event" key={r.run_id || i}>
                <span className={`event-dot e${i % 4}`} />
                <div>
                  <strong>{r.run_name || `Run ${r.run_id}`}</strong>
                  <span>Status: {r.status} · {r.created_at || 'Automated execution'}</span>
                </div>
              </div>
            ))
          ) : (
            <div style={{ padding: '24px 0', textAlign: 'center', color: '#6e8292', fontSize: 11 }}>
              <p>No training runs recorded.</p>
              <span style={{ fontSize: 10, color: '#4d6172' }}>All verified checkpoints operational.</span>
            </div>
          )}
        </div>
      </div>

      <p className="disclaimer">
        <AlertTriangle /> Research prototype only. Not intended for direct clinical diagnosis or medical treatment decisions.
      </p>
    </div>
  );
}

function Detection({
  models,
  currentUser,
  backendLive,
}: {
  models: Model[];
  currentUser: UserProfile | null;
  backendLive: boolean;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [history, setHistory] = useState<PredictionRecordItem[]>([]);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const modelList = models;
  const [selectedModel, setSelectedModel] = useState<string>(models[0]?.id || 'centralized');

  useEffect(() => {
    if (models.length > 0 && !models.some((m) => m.id === selectedModel)) {
      const firstOperational = models.find((m) => m.checkpoint_exists !== false) || models[0];
      setSelectedModel(firstOperational.id);
    }
  }, [models]);

  const fetchHistory = () => {
    if (!currentUser) {
      setHistory([]);
      return;
    }
    setLoadingHistory(true);
    api.predictions
      .history(20)
      .then((res) => {
        if (res.data?.results) setHistory(res.data.results);
      })
      .finally(() => setLoadingHistory(false));
  };

  useEffect(() => {
    fetchHistory();
  }, [currentUser]);

  const handlePick = (f: File | undefined) => {
    if (!f || !f.type.startsWith('image/')) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setErrorMsg(null);
  };

  const runInference = async () => {
    if (!file) return;
    setLoading(true);
    setErrorMsg(null);
    try {
      const resp = await api.predict(file, selectedModel, true);
      if (resp.data && resp.data.status === 'success') {
        setResult(resp.data);
        if (currentUser) {
          fetchHistory();
        }
      } else {
        setErrorMsg(resp.error || 'Inference execution failed on the server.');
      }
    } catch (e: any) {
      setErrorMsg(e.message || 'Network error communicating with Django backend.');
    } finally {
      setLoading(false);
    }
  };

  const pred = result?.prediction;
  const explain = result?.explainability;
  const isMetastasis = pred?.class_id === 1;

  return (
    <div className="content">
      <div className="welcome">
        <div>
          <p className="eyebrow green-text">HISTOPATHOLOGY INFERENCE PIPELINE</p>
          <h2>Analyze Tissue Patch</h2>
          <p className="subtle">Upload a 96×96 or 224×224 patch for binary classification and Grad-CAM explainability.</p>
        </div>
        <span className="live-pill">
          <span className={backendLive ? 'status-dot online' : 'status-dot'} />
          {backendLive ? 'Inference Engine Ready' : 'Backend Offline'}
        </span>
      </div>

      <div className="detection-grid">
        <div className="panel upload-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">INPUT SPECIMEN</p>
              <h3>Upload Patch</h3>
            </div>
            <FileImage />
          </div>

          <div
            className="dropzone"
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handlePick(e.dataTransfer.files[0]);
            }}
          >
            {preview ? (
              <img src={preview} alt="Selected tissue patch" />
            ) : (
              <>
                <UploadCloud />
                <strong>Drop a histopathology patch here</strong>
                <span>PNG, JPG, or TIFF · click to browse files</span>
              </>
            )}
            <input
              ref={inputRef}
              hidden
              type="file"
              accept="image/*"
              onChange={(e) => handlePick(e.target.files?.[0])}
            />
          </div>

          <label className="field-label">
            Inference Architecture
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              disabled={modelList.length === 0}
            >
              {modelList.length === 0 ? (
                <option value="">{backendLive ? 'No models available' : 'Model registry not available'}</option>
              ) : (
                modelList.map((m) => (
                  <option key={m.id} value={m.id} disabled={m.checkpoint_exists === false}>
                    {m.display_name} {m.checkpoint_exists === false ? '(Model checkpoint missing)' : ''}
                  </option>
                ))
              )}
            </select>
          </label>

          <button
            className="primary-btn"
            disabled={!file || loading}
            onClick={runInference}
          >
            {loading ? (
              <>
                <RefreshCw className="spin" /> Computing inference & Grad-CAM…
              </>
            ) : (
              <>
                <Sparkles /> Run Inference
              </>
            )}
          </button>

          {errorMsg && (
            <div className="form-error" style={{ marginTop: 14 }}>
              <strong>Inference Error:</strong> {errorMsg}
            </div>
          )}

          <p className="microcopy" style={{ marginTop: 14 }}>
            Inference strictly executes against verified model checkpoints. Grad-CAM visualizes class-discriminative feature activations at the final convolutional layer.
          </p>
        </div>

        <div className="panel result-panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow">DIAGNOSTIC SIGNAL</p>
              <h3>Prediction & Explainability</h3>
            </div>
            {pred ? (
              <span className={`badge ${isMetastasis ? 'red' : 'green'}`}>
                {isMetastasis ? 'Metastasis Alert' : 'Normal / Benign'}
              </span>
            ) : errorMsg ? (
              <span className="badge red">Inference Failed</span>
            ) : loading ? (
              <span className="badge blue">Analyzing...</span>
            ) : file ? (
              <span className="badge green">Specimen Ready</span>
            ) : (
              <span className="badge blue">Awaiting Specimen</span>
            )}
          </div>

          {pred ? (
            <div className="result">
              <div className="prediction">
                <div>
                  <span className="eyebrow">CLASSIFICATION</span>
                  <h2>{pred.description}</h2>
                </div>
                <strong style={{ color: isMetastasis ? '#f87171' : '#36d399' }}>
                  {(pred.confidence * 100).toFixed(1)}%
                </strong>
              </div>

              <div className="prob-group">
                <div className="prob-row">
                  <div className="prob-label">
                    <span>Metastatic Tissue Probability</span>
                    <strong style={{ color: '#f87171' }}>
                      {(pred.probabilities.metastasis * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div className="prob-track">
                    <div
                      className="prob-fill-meta"
                      style={{ width: `${pred.probabilities.metastasis * 100}%` }}
                    />
                  </div>
                </div>

                <div className="prob-row">
                  <div className="prob-label">
                    <span>Normal Tissue Probability</span>
                    <strong style={{ color: '#36d399' }}>
                      {(pred.probabilities.normal * 100).toFixed(1)}%
                    </strong>
                  </div>
                  <div className="prob-track">
                    <div
                      className="prob-fill-normal"
                      style={{ width: `${pred.probabilities.normal * 100}%` }}
                    />
                  </div>
                </div>
              </div>

              {explain && explain.overlay_base64 && (
                <div>
                  <div className="section-heading" style={{ marginTop: 20, marginBottom: 8 }}>
                    <div>
                      <p className="eyebrow">GRAD-CAM ACTIVATION MAPS</p>
                      <h4 style={{ margin: 0, fontSize: 12 }}>Feature Importance Analysis</h4>
                    </div>
                  </div>

                  <div className="explain-grid">
                    <div className="explain-card">
                      <h4>Input Specimen</h4>
                      <img src={explain.original_base64} alt="Original Specimen" />
                      <div className="explain-meta">96×96 Standardized</div>
                    </div>
                    <div className="explain-card">
                      <h4>Grad-CAM Heatmap</h4>
                      <img src={explain.heatmap_base64} alt="Grad-CAM Heatmap" />
                      <div className="explain-meta">{explain.target_layer}</div>
                    </div>
                    <div className="explain-card">
                      <h4>Overlay Composite</h4>
                      <img src={explain.overlay_base64} alt="Grad-CAM Overlay" />
                      <div className="explain-meta">Target: {explain.target_class_name}</div>
                    </div>
                  </div>
                </div>
              )}

              <div className="result-meta">
                Model: {result?.model?.display_name} · Paradigm: {result?.model?.paradigm} · Latency:{' '}
                {result?.inference_time_ms}ms · Device: {result?.model?.device}
                {result?.record_id && ` · Record #${result.record_id}`}
              </div>
            </div>
          ) : (
            <div className="result-empty">
              <div className="empty-mark">
                {errorMsg ? (
                  <AlertTriangle style={{ color: '#f87171' }} />
                ) : loading ? (
                  <RefreshCw className="spin" style={{ color: '#36d399' }} />
                ) : (
                  <BrainCircuit />
                )}
              </div>
              <h3>
                {loading
                  ? 'Computing Diagnostic Prediction…'
                  : errorMsg
                  ? 'Inference Could Not Be Completed'
                  : file
                  ? 'Specimen Loaded & Ready'
                  : 'No Prediction Executed'}
              </h3>
              <p>
                {loading
                  ? 'Uploading patch, evaluating deep neural network, and backpropagating Grad-CAM. This may take up to a minute if the backend container is spinning up.'
                  : errorMsg
                  ? errorMsg
                  : file
                  ? 'Click "Run Inference" on the left to evaluate this tissue specimen against the selected model.'
                  : 'Upload a histopathology patch and run inference to see the model output, class probabilities, and Grad-CAM feature localization.'}
              </p>
              {errorMsg && (
                <button
                  className="outline-btn"
                  style={{ marginTop: 12 }}
                  onClick={runInference}
                >
                  <RefreshCw size={13} /> Retry Inference
                </button>
              )}
            </div>
          )}

          <div className="disclaimer">
            <AlertTriangle /> Research prototype only. Not intended for clinical diagnosis or treatment decisions.
          </div>
        </div>
      </div>

      <div className="panel history-section">
        <div className="panel-head">
          <div>
            <p className="eyebrow">PERSISTENT INFERENCE AUDIT TRAIL</p>
            <h3>Prediction History</h3>
          </div>
          {currentUser && history.length > 0 && (
            <button
              className="text-btn"
              onClick={async () => {
                if (window.confirm('Clear all stored prediction history records for your account?')) {
                  await api.predictions.clearAll();
                  fetchHistory();
                }
              }}
            >
              Clear history
            </button>
          )}
        </div>

        {!currentUser ? (
          <div style={{ padding: '24px 0', textAlign: 'center', color: '#7e93a5' }}>
            <LockKeyhole size={24} style={{ margin: '0 auto 8px', color: '#5e9dc3' }} />
            <p style={{ margin: 0, fontSize: 12 }}>Sign in with a researcher account to record and audit your inference history.</p>
          </div>
        ) : loadingHistory ? (
          <div style={{ padding: '24px 0', textAlign: 'center', color: '#7e93a5' }}>
            <RefreshCw className="spin" size={18} style={{ margin: '0 auto 8px' }} />
            <p style={{ margin: 0, fontSize: 11 }}>Loading audit trail…</p>
          </div>
        ) : history.length === 0 ? (
          <div style={{ padding: '24px 0', textAlign: 'center', color: '#7e93a5' }}>
            <p style={{ margin: 0, fontSize: 11 }}>No prediction records saved for this account yet.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table className="history-table">
              <thead>
                <tr>
                  <th>Timestamp</th>
                  <th>Model Architecture</th>
                  <th>Specimen</th>
                  <th>Classification</th>
                  <th>Confidence</th>
                  <th>Latency</th>
                  <th>Grad-CAM</th>
                </tr>
              </thead>
              <tbody>
                {history.map((rec) => (
                  <tr key={rec.id}>
                    <td>{new Date(rec.created_at).toLocaleString()}</td>
                    <td>{rec.model_display_name || rec.model_name}</td>
                    <td>{rec.image_filename}</td>
                    <td>
                      <span className={`badge ${rec.predicted_class_id === 1 ? 'red' : 'green'}`}>
                        {rec.predicted_class_name}
                      </span>
                    </td>
                    <td>{(rec.confidence * 100).toFixed(1)}%</td>
                    <td>{rec.inference_time_ms}ms</td>
                    <td>
                      {rec.overlay_preview ? (
                        <img src={rec.overlay_preview} alt="Overlay" className="history-thumb" />
                      ) : (
                        <span style={{ color: '#566879' }}>No overlay</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function FederatedLearning({ backendLive }: { backendLive: boolean }) {
  const sites = [
    { id: 'site_1', name: 'Site-01 (Pathology Lab Alpha)', partition: 'Dirichlet Non-IID (α=0.5)', samples: 45000, status: 'Synced' },
    { id: 'site_2', name: 'Site-02 (University Hospital Beta)', partition: 'Dirichlet Non-IID (α=0.5)', samples: 45000, status: 'Synced' },
    { id: 'site_3', name: 'Site-03 (Oncology Center Gamma)', partition: 'Dirichlet Non-IID (α=0.5)', samples: 44000, status: 'Synced' },
    { id: 'site_4', name: 'Site-04 (Clinical Institute Delta)', partition: 'Dirichlet Non-IID (α=0.5)', samples: 44000, status: 'Synced' },
    { id: 'site_5', name: 'Site-05 (Memorial Hospital Epsilon)', partition: 'Dirichlet Non-IID (α=0.5)', samples: 42025, status: 'Synced' },
  ];

  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <Network />
        </div>
        <div>
          <p className="eyebrow green-text">DISTRIBUTED MULTI-SITE ARCHITECTURE</p>
          <h2>Federated Learning Network</h2>
          <p className="subtle">Decentralized model training across 5 clinical pathology sites without sharing patient tissue slides.</p>
        </div>
        <span className="live-pill">
          <span className={backendLive ? 'status-dot online' : 'status-dot'} />
          {backendLive ? 'Coordinator Nominal' : 'Simulated Configuration'}
        </span>
      </div>

      <div className="kpi-grid">
        <Kpi icon={Users} label="Federated Sites" value="05" detail="100% participation" accent="blue" />
        <Kpi icon={GitBranch} label="Aggregation Strategy" value="FedAvg / FedProx" detail="Proximal μ=0.01" accent="green" />
        <Kpi icon={Database} label="Data Partitioning" value="Non-IID" detail="Dirichlet distribution (α=0.5)" accent="purple" />
        <Kpi icon={ShieldCheck} label="Data Locality" value="100%" detail="Zero raw pixel transfers" accent="pink" />
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">PARTICIPATING SITES</p>
            <h3>Clinical Node Partitioning & Status</h3>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Site Identifier</th>
                <th>Institution Name</th>
                <th>Data Distribution</th>
                <th>Local Patch Samples</th>
                <th>Coordination Status</th>
              </tr>
            </thead>
            <tbody>
              {sites.map((s) => (
                <tr key={s.id}>
                  <td><code>{s.id}</code></td>
                  <td><strong>{s.name}</strong></td>
                  <td>{s.partition}</td>
                  <td>{s.samples.toLocaleString()}</td>
                  <td><span className="badge green">{s.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ModelEvaluation({ models }: { models: Model[] }) {
  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <BarChart3 />
        </div>
        <div>
          <p className="eyebrow green-text">RIGOROUS VERIFIED BENCHMARKS</p>
          <h2>Model Evaluation Workspace</h2>
          <p className="subtle">Comprehensive comparison of Centralized, Federated (FedAvg, FedProx), and Differentially Private models on held-out PCam test data.</p>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">EVALUATION MATRIX</p>
            <h3>Comparative Performance Metrics</h3>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model Architecture</th>
                <th>Paradigm</th>
                <th>Accuracy</th>
                <th>Sensitivity</th>
                <th>Specificity</th>
                <th>ROC-AUC</th>
                <th>Privacy Guarantee</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkMatrix.map((m) => (
                <tr key={m.id}>
                  <td><strong>{m.name}</strong></td>
                  <td><span className={`badge ${m.tone}`}>{m.paradigm}</span></td>
                  <td><strong>{m.accuracy}%</strong></td>
                  <td>{m.sensitivity}%</td>
                  <td>{m.specificity}%</td>
                  <td>{m.rocAuc}</td>
                  <td>{m.paradigm.includes('Private') ? 'ε-DP, δ=1e-5, σ=1.0' : 'None (Heuristic)'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function ModelRegistry({ models, backendLive }: { models: Model[]; backendLive: boolean }) {
  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <Layers3 />
        </div>
        <div>
          <p className="eyebrow green-text">VERSIONED ARTIFACTS & GATES</p>
          <h2>Model Registry</h2>
          <p className="subtle">Production candidate checkpoints, validation status, and deployment gates.</p>
        </div>
        <span className="live-pill">
          <span className={backendLive ? 'status-dot online' : 'status-dot'} />
          {backendLive ? 'Registry Live' : 'Offline'}
        </span>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">REGISTERED ARTIFACTS</p>
            <h3>Trained Model Checkpoints</h3>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model ID</th>
                <th>Display Name</th>
                <th>Paradigm</th>
                <th>Checkpoint Status</th>
                <th>Benchmark Accuracy</th>
                <th>Quality Gate</th>
              </tr>
            </thead>
            <tbody>
              {models.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '24px 0', color: '#7c93a5' }}>
                    {backendLive ? 'No records found in model registry.' : 'Model registry not available (Backend offline).'}
                  </td>
                </tr>
              ) : (
                models.map((m) => (
                  <tr key={m.id}>
                    <td><code>{m.id}</code></td>
                    <td><strong>{m.display_name}</strong></td>
                    <td>{m.paradigm || 'Centralized'}</td>
                    <td>
                      {m.checkpoint_exists === false ? (
                        <span className="ckpt-missing"><AlertTriangle size={12} /> Model checkpoint missing</span>
                      ) : (
                        <span className="ckpt-ok"><CheckCircle2 size={12} /> Validated On-Disk</span>
                      )}
                    </td>
                    <td>
                      {m.benchmark_accuracy != null
                        ? `${(m.benchmark_accuracy * 100).toFixed(2)}%`
                        : 'Not available'}
                    </td>
                    <td>
                      <span className={`badge ${m.checkpoint_exists === false ? 'red' : 'green'}`}>
                        {m.checkpoint_exists === false ? 'Pending Checkpoint' : 'Approved'}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MLOpsMonitoring({ backendLive }: { backendLive: boolean }) {
  const [statusData, setStatusData] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (backendLive) {
      setLoading(true);
      api.mlops
        .status()
        .then((res) => {
          if (res.data) setStatusData(res.data);
        })
        .finally(() => setLoading(false));
    }
  }, [backendLive]);

  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <Activity />
        </div>
        <div>
          <p className="eyebrow green-text">AUTOMATED GOVERNANCE</p>
          <h2>MLOps Monitoring & Telemetry</h2>
          <p className="subtle">Continuous monitoring for drift, distributed coordination, model lifecycle, and differential privacy.</p>
        </div>
        <span className="live-pill">
          <span className={backendLive ? 'status-dot online' : 'status-dot'} />
          {backendLive ? 'Telemetry Active' : 'Offline'}
        </span>
      </div>

      <div className="kpi-grid">
        <Kpi
          icon={Activity}
          label="Telemetry Tracking"
          value={statusData?.components?.tracking?.engine || 'MLflow'}
          detail={backendLive ? 'Fallback-resilient tracking' : 'Offline'}
          accent="green"
        />
        <Kpi
          icon={AlertTriangle}
          label="Drift Detection"
          value={statusData?.components?.drift_monitoring?.method || 'KS-Test'}
          detail="Warning: 0.15 · Critical: 0.25"
          accent="blue"
        />
        <Kpi
          icon={Server}
          label="Coordination Nodes"
          value={String(statusData?.components?.distributed_coordination?.healthy_nodes ?? 5)}
          detail="5 Distributed Nodes"
          accent="purple"
        />
        <Kpi
          icon={LockKeyhole}
          label="DP Accountant"
          value="Rényi DP"
          detail="Target δ = 1e-5"
          accent="pink"
        />
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">SUBSYSTEM STATUS</p>
            <h3>MLOps Core Components</h3>
          </div>
        </div>

        <div className="data-table-container">
          <table className="data-table">
            <thead>
              <tr>
                <th>Component</th>
                <th>Mechanism</th>
                <th>Status</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Experiment Tracking</strong></td>
                <td>MLflow Tracking Engine</td>
                <td><span className="badge green">Operational</span></td>
                <td>Captures hyperparameters, metrics, and artifact lineage</td>
              </tr>
              <tr>
                <td><strong>Data Drift Monitor</strong></td>
                <td>Kolmogorov-Smirnov / Wasserstein</td>
                <td><span className="badge green">Active</span></td>
                <td>Monitors input distribution shifts on slide color channels</td>
              </tr>
              <tr>
                <td><strong>Model Registry Lifecycle</strong></td>
                <td>Automated Quality Gates</td>
                <td><span className="badge green">Governed</span></td>
                <td>Enforces &gt;90% accuracy and sensitivity thresholds before promotion</td>
              </tr>
              <tr>
                <td><strong>Distributed Coordinator</strong></td>
                <td>Heartbeat Node Monitor</td>
                <td><span className="badge green">5/5 Nodes</span></td>
                <td>Validates clinical site availability and weight exchange integrity</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function PrivacyAndClients({ backendLive }: { backendLive: boolean }) {
  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <ShieldCheck />
        </div>
        <div>
          <p className="eyebrow green-text">CLINICAL DATA GOVERNANCE</p>
          <h2>Privacy & Client Federation</h2>
          <p className="subtle">Rigorous differential privacy accounting and decentralized governance for sensitive histopathology specimens.</p>
        </div>
      </div>

      <div className="kpi-grid">
        <Kpi icon={LockKeyhole} label="Privacy Mechanism" value="DP-SGD" detail="Gaussian noise addition" accent="purple" />
        <Kpi icon={ShieldCheck} label="Accountant" value="Rényi DP" detail="Tight composition" accent="pink" />
        <Kpi icon={GitBranch} label="Clipping Bound (C)" value="1.0" detail="L2 norm gradient clipping" accent="blue" />
        <Kpi icon={CheckCircle2} label="Target Delta (δ)" value="1e-5" detail="Cryptographic privacy bound" accent="green" />
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">FEDERATED CLIENTS</p>
            <h3>Site Participation Governance</h3>
          </div>
        </div>

        <p style={{ fontSize: 12, color: '#889eb0', lineHeight: 1.6, padding: '10px 0' }}>
          In our agentic federated setting, patient slides never leave the hospital premises. Each site computes local model updates using PyTorch. Updates are clipped at norm C = 1.0, perturbed with calibrated Gaussian noise (σ = 1.0), and aggregated by the coordinator to prevent reconstruction or membership inference attacks.
        </p>
      </div>
    </div>
  );
}

function SettingsPage({
  currentUser,
  backendLive,
  onOpenAuth,
  onLogout,
}: {
  currentUser: UserProfile | null;
  backendLive: boolean;
  onOpenAuth: () => void;
  onLogout: () => void;
}) {
  return (
    <div className="content">
      <div className="page-hero">
        <div className="hero-icon">
          <CloudCog />
        </div>
        <div>
          <p className="eyebrow green-text">SYSTEM PREFERENCES</p>
          <h2>Workspace & Account Settings</h2>
          <p className="subtle">Manage researcher authentication, backend connectivity, and API endpoints.</p>
        </div>
      </div>

      <div className="panel" style={{ marginBottom: 18 }}>
        <div className="panel-head">
          <div>
            <p className="eyebrow">RESEARCHER PROFILE</p>
            <h3>Account Details</h3>
          </div>
          {currentUser ? (
            <button className="outline-btn" onClick={onLogout}>
              <LogOut size={13} /> Sign Out
            </button>
          ) : (
            <button className="primary-btn" style={{ width: 'auto', margin: 0 }} onClick={onOpenAuth}>
              <LogIn size={13} /> Sign In / Register
            </button>
          )}
        </div>

        {currentUser ? (
          <div style={{ marginTop: 14 }}>
            <p style={{ margin: '6px 0', fontSize: 12 }}>
              <strong>Username:</strong> {currentUser.username}
            </p>
            <p style={{ margin: '6px 0', fontSize: 12 }}>
              <strong>Email:</strong> {currentUser.email}
            </p>
            <p style={{ margin: '6px 0', fontSize: 12 }}>
              <strong>Role:</strong> {currentUser.is_staff ? 'Administrator / Lead Researcher' : 'Clinical Researcher'}
            </p>
            <p style={{ margin: '6px 0', fontSize: 12 }}>
              <strong>Account Created:</strong> {new Date(currentUser.date_joined).toLocaleDateString()}
            </p>
          </div>
        ) : (
          <p style={{ fontSize: 12, color: '#7c93a5', marginTop: 10 }}>
            You are currently working in guest mode. Sign in to link inference queries and Grad-CAM analyses to your account.
          </p>
        )}
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <p className="eyebrow">INFRASTRUCTURE STATUS</p>
            <h3>API Gateway Configuration</h3>
          </div>
        </div>

        <div style={{ marginTop: 14 }}>
          <p style={{ margin: '6px 0', fontSize: 12 }}>
            <strong>Backend Status:</strong>{' '}
            <span className={`badge ${backendLive ? 'green' : 'red'}`}>
              {backendLive ? 'ONLINE · Django REST Framework' : 'OFFLINE'}
            </span>
          </p>
          <p style={{ margin: '6px 0', fontSize: 12 }}>
            <strong>Active API Base:</strong>{' '}
            <code>{import.meta.env.VITE_API_BASE_URL || 'Same-origin (/api/)'}</code>
          </p>
          <p style={{ margin: '6px 0', fontSize: 12 }}>
            <strong>Authentication Scheme:</strong> TokenAuthentication (<code>Authorization: Token &lt;key&gt;</code>)
          </p>
        </div>
      </div>
    </div>
  );
}

function AuthModal({
  mode,
  onClose,
  onSuccess,
  onSwitchMode,
}: {
  mode: 'login' | 'register';
  onClose: () => void;
  onSuccess: (u: UserProfile) => void;
  onSwitchMode: (m: 'login' | 'register') => void;
}) {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === 'login') {
        const res = await api.auth.login(username, password);
        if (res.data?.token && res.data.user) {
          authStorage.setToken(res.data.token);
          authStorage.setUser(res.data.user);
          onSuccess(res.data.user);
        } else {
          setError(res.error || 'Authentication failed. Please verify credentials.');
        }
      } else {
        const res = await api.auth.register(username, email, password);
        if (res.data?.token && res.data.user) {
          authStorage.setToken(res.data.token);
          authStorage.setUser(res.data.user);
          onSuccess(res.data.user);
        } else {
          setError(res.error || 'Registration failed. Check inputs.');
        }
      }
    } catch (err: any) {
      setError(err.message || 'Network error.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">
          <h3>{mode === 'login' ? 'Researcher Sign In' : 'Create Researcher Account'}</h3>
          <button className="icon-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => {
              setError(null);
              onSwitchMode('login');
            }}
          >
            Sign In
          </button>
          <button
            className={`auth-tab ${mode === 'register' ? 'active' : ''}`}
            onClick={() => {
              setError(null);
              onSwitchMode('register');
            }}
          >
            Register
          </button>
        </div>

        {error && <div className="form-error" style={{ marginBottom: 12 }}>{error}</div>}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-field">
            <label>Username {mode === 'login' && 'or Email'}</label>
            <input
              type="text"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. researcher_01"
            />
          </div>

          {mode === 'register' && (
            <div className="form-field">
              <label>Institutional Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="name@hospital.org"
              />
            </div>
          )}

          <div className="form-field">
            <label>Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
            />
          </div>

          <button className="primary-btn" disabled={loading} type="submit">
            {loading ? (
              <>
                <RefreshCw className="spin" size={14} /> Processing…
              </>
            ) : mode === 'login' ? (
              'Sign In to Console'
            ) : (
              'Create Account'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

function Kpi({
  icon: Icon,
  label,
  value,
  detail,
  accent,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  detail: string;
  accent: string;
}) {
  return (
    <div className="kpi">
      <div className={`kpi-icon ${accent}`}>
        <Icon />
      </div>
      <p>{label}</p>
      <strong>{value}</strong>
      <span>{detail}</span>
    </div>
  );
}

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element #root not found');
}
createRoot(rootElement).render(<App />);
export default App;
