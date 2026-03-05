import { useState, type KeyboardEvent } from 'react'

interface SearchResult {
  rank: number
  score: number
  image_url: string
  name: string
  embedding_index: number
  rgb: [number, number, number] | null
}

interface SearchResponse {
  query: string
  elapsed_ms: number
  results: SearchResult[]
}

interface RefineResponse {
  results: SearchResult[]
  elapsed_ms: number
}

export default function App() {
  const [query, setQuery] = useState('')
  const [mode, setMode] = useState<'image' | 'text'>('image')
  const [dataset, setDataset] = useState<'color' | 'sketchy_test'>('color')
  const [results, setResults] = useState<SearchResult[]>([])
  const [elapsed, setElapsed] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentQuery, setCurrentQuery] = useState('')

  const [positiveSet, setPositiveSet] = useState<Set<number>>(new Set())
  const [negativeSet, setNegativeSet] = useState<Set<number>>(new Set())

  function resetFeedback() {
    setPositiveSet(new Set())
    setNegativeSet(new Set())
  }

  async function search() {
    const q = query.trim()
    if (!q) return

    setLoading(true)
    setError(null)
    resetFeedback()

    try {
      const res = await fetch(`/search?query=${encodeURIComponent(q)}&top_k=20&mode=${mode}&dataset=${dataset}`)
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data: SearchResponse = await res.json()
      setResults(data.results)
      setElapsed(data.elapsed_ms)
      setCurrentQuery(q)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  async function refine() {
    if (!positiveSet.size && !negativeSet.size) return
    setLoading(true)
    setError(null)

    try {
      const res = await fetch('/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: currentQuery,
          mode,
          dataset,
          positive_indices: [...positiveSet],
          negative_indices: [...negativeSet],
        }),
      })
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data: RefineResponse = await res.json()
      setResults(data.results)
      setElapsed(data.elapsed_ms)
      resetFeedback()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  function toggleFeedback(idx: number, sign: 'positive' | 'negative') {
    if (sign === 'positive') {
      setPositiveSet(prev => {
        const next = new Set(prev)
        if (next.has(idx)) { next.delete(idx); return next }
        setNegativeSet(neg => { const n = new Set(neg); n.delete(idx); return n })
        next.add(idx)
        return next
      })
    } else {
      setNegativeSet(prev => {
        const next = new Set(prev)
        if (next.has(idx)) { next.delete(idx); return next }
        setPositiveSet(pos => { const n = new Set(pos); n.delete(idx); return n })
        next.add(idx)
        return next
      })
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') search()
  }

  const hasFeedback = positiveSet.size > 0 || negativeSet.size > 0

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">

        <h1 className="text-2xl font-semibold text-gray-800 mb-6">
          Color Image Retrieval
        </h1>

        {/* Search bar */}
        <div className="flex gap-3 mb-6">
          <select
            value={dataset}
            onChange={e => {
              setDataset(e.target.value as 'color' | 'sketchy_test')
              setResults([])
              setElapsed(null)
              resetFeedback()
            }}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="color">Color</option>
            <option value="sketchy_test">Sketchy</option>
          </select>
          <select
            value={mode}
            onChange={e => setMode(e.target.value as 'image' | 'text')}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="image">Image</option>
            <option value="text">Text</option>
          </select>
          <input
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder="Search by color or description..."
            className="flex-1 border border-gray-300 rounded-lg px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={search}
            disabled={loading}
            className="bg-blue-600 text-white px-5 py-2 rounded-lg text-sm font-medium hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {loading ? 'Searching…' : 'Search'}
          </button>
        </div>

        {/* Status line + Refine button */}
        {elapsed !== null && !loading && (
          <div className="flex items-center justify-between mb-4">
            <p className="text-xs text-gray-400">
              {results.length} results in {elapsed.toFixed(1)} ms
              {hasFeedback && (
                <span className="ml-2">
                  — {positiveSet.size} positive, {negativeSet.size} negative selected
                </span>
              )}
            </p>
            {hasFeedback && (
              <button
                onClick={refine}
                disabled={loading}
                className="bg-indigo-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 transition-colors"
              >
                Refine
              </button>
            )}
          </div>
        )}

        {/* Error */}
        {error && (
          <p className="text-sm text-red-500 mb-4">{error}</p>
        )}

        {/* Image grid */}
        <div className="grid grid-cols-4 gap-4">
          {results.map(r => {
            const isPos = positiveSet.has(r.embedding_index)
            const isNeg = negativeSet.has(r.embedding_index)
            const ringClass = isPos
              ? 'ring-2 ring-green-500'
              : isNeg
              ? 'ring-2 ring-red-500'
              : 'border border-gray-100'

            return (
              <div key={r.embedding_index} className={`bg-white rounded-lg overflow-hidden shadow-sm ${ringClass}`}>
                <div className="relative">
                  <img
                    src={r.image_url}
                    alt={r.name}
                    className="w-full aspect-square object-cover"
                  />
                  <div className="absolute top-1 right-1 flex gap-1">
                    <button
                      onClick={() => toggleFeedback(r.embedding_index, 'positive')}
                      title="Relevant"
                      className={`w-7 h-7 rounded-full text-sm font-bold shadow transition-colors ${
                        isPos
                          ? 'bg-green-500 text-white'
                          : 'bg-white/80 text-green-700 hover:bg-green-100'
                      }`}
                    >
                      +
                    </button>
                    <button
                      onClick={() => toggleFeedback(r.embedding_index, 'negative')}
                      title="Not relevant"
                      className={`w-7 h-7 rounded-full text-sm font-bold shadow transition-colors ${
                        isNeg
                          ? 'bg-red-500 text-white'
                          : 'bg-white/80 text-red-700 hover:bg-red-100'
                      }`}
                    >
                      −
                    </button>
                  </div>
                </div>
                <div className="p-2 flex items-end justify-between gap-1">
                  <div className="min-w-0">
                    <p className="text-xs font-medium text-gray-700 truncate">{r.name}</p>
                    <p className="text-xs text-gray-400">{r.score.toFixed(3)}</p>
                  </div>
                  {r.rgb && (
                    <p className="text-xs text-gray-400 shrink-0">rgb({r.rgb.join(', ')})</p>
                  )}
                </div>
              </div>
            )
          })}
        </div>

      </div>
    </div>
  )
}
