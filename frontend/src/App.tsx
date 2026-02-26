import { useState, type KeyboardEvent } from 'react'

interface SearchResult {
  rank: number
  score: number
  image_url: string
  name: string
}

interface SearchResponse {
  query: string
  elapsed_ms: number
  results: SearchResult[]
}

export default function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<SearchResult[]>([])
  const [elapsed, setElapsed] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function search() {
    const q = query.trim()
    if (!q) return

    setLoading(true)
    setError(null)

    try {
      const res = await fetch(`/search?query=${encodeURIComponent(q)}&top_k=20`)
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data: SearchResponse = await res.json()
      setResults(data.results)
      setElapsed(data.elapsed_ms)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setResults([])
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') search()
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">

        <h1 className="text-2xl font-semibold text-gray-800 mb-6">
          Color Image Retrieval
        </h1>

        {/* Search bar */}
        <div className="flex gap-3 mb-6">
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

        {/* Status line */}
        {elapsed !== null && !loading && (
          <p className="text-xs text-gray-400 mb-4">
            {results.length} results in {elapsed.toFixed(1)} ms
          </p>
        )}

        {/* Error */}
        {error && (
          <p className="text-sm text-red-500 mb-4">{error}</p>
        )}

        {/* Image grid */}
        <div className="grid grid-cols-4 gap-4">
          {results.map(r => (
            <div key={r.rank} className="bg-white rounded-lg overflow-hidden shadow-sm border border-gray-100">
              <img
                src={r.image_url}
                alt={r.name}
                className="w-full aspect-square object-cover"
              />
              <div className="p-2">
                <p className="text-xs font-medium text-gray-700 truncate">{r.name}</p>
                <p className="text-xs text-gray-400">{r.score.toFixed(3)}</p>
              </div>
            </div>
          ))}
        </div>

      </div>
    </div>
  )
}
