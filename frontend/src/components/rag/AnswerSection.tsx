'use client'

import { useEffect, useRef } from 'react'
import { marked } from 'marked'
import katex from 'katex'
import hljs from 'highlight.js/lib/core'
import javascript from 'highlight.js/lib/languages/javascript'
import python from 'highlight.js/lib/languages/python'
import bash from 'highlight.js/lib/languages/bash'
import 'highlight.js/styles/github.min.css'
import 'katex/dist/katex.min.css'

// Register languages
hljs.registerLanguage('javascript', javascript)
hljs.registerLanguage('python', python)
hljs.registerLanguage('bash', bash)

// The system prompt asks the model for LaTeX ($inline$ and $$block$$), so answers
// arrive with real notation in them. Markdown alone leaves it as literal text.
//
// Math is pulled out BEFORE the markdown pass rather than typeset after it,
// because subscripts collide with markdown: "L_1 ... L_2" on one line is read as
// emphasis, and marked would eat the underscores and wrap the middle in <em>
// before KaTeX ever saw it.
type MathNode = { tex: string; display: boolean }

const placeholder = (i: number) => `@@KATEX_BLOCK_${i}@@`

function extractMath(source: string): { text: string; nodes: MathNode[] } {
  const nodes: MathNode[] = []

  const take = (tex: string, display: boolean) => {
    nodes.push({ tex: tex.trim(), display })
    return placeholder(nodes.length - 1)
  }

  // $$...$$ first, so the inline pass cannot split a block delimiter in half.
  let text = source.replace(/\$\$([\s\S]+?)\$\$/g, (_match, tex: string) =>
    take(tex, true),
  )

  // $...$ on a single line. Two dollar amounts in one sentence ("costs $7 per
  // month and $12 with backups") otherwise look exactly like one math span, so
  // claim it only when it carries a TeX control character, or is a short
  // unbroken token like $N$ or $x_i$.
  const looksLikeTex = (tex: string) =>
    /[\\^_{}]/.test(tex) || (!/\s/.test(tex) && tex.length <= 12)

  text = text.replace(/\$([^$\n]+?)\$/g, (match, tex: string) =>
    looksLikeTex(tex) ? take(tex, false) : match,
  )

  return { text, nodes }
}

function restoreMath(html: string, nodes: MathNode[]): string {
  return nodes.reduce((acc, node, i) => {
    let rendered: string
    try {
      rendered = katex.renderToString(node.tex, {
        displayMode: node.display,
        throwOnError: false,
        output: 'html',
      })
    } catch {
      // Malformed LaTeX should degrade to the original text, not blank the answer.
      rendered = node.display ? `$$${node.tex}$$` : `$${node.tex}$`
    }
    // Function form: KaTeX output can contain $ sequences that string replacement
    // would otherwise treat as capture-group references.
    return acc.replace(placeholder(i), () => rendered)
  }, html)
}

interface AnswerSectionProps {
  answer: string
  processingTime?: number
  isVisible: boolean
}

export function AnswerSection({ answer, processingTime, isVisible }: AnswerSectionProps) {
  const contentRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (contentRef.current && answer) {
      // Render markdown (marked.parse can be sync or async depending on version)
      const renderMarkdown = async () => {
        try {
          const { text, nodes } = extractMath(answer)
          const parsed = (await marked.parse(text)) as string
          const html = restoreMath(parsed, nodes)
          if (contentRef.current) {
            contentRef.current.innerHTML = html
            
            // Highlight code blocks after a brief delay to ensure DOM is updated
            setTimeout(() => {
              if (contentRef.current) {
                contentRef.current.querySelectorAll('pre code').forEach((block) => {
                  hljs.highlightElement(block as HTMLElement)
                })
              }
            }, 0)
          }
        } catch (error) {
          console.error('Error parsing markdown:', error)
          if (contentRef.current) {
            contentRef.current.innerHTML = answer
          }
        }
      }
      
      renderMarkdown()
    }
  }, [answer])

  if (!isVisible) return null

  return (
    <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-6 md:px-8 py-5 border-b border-gray-100 bg-gradient-to-r from-gray-50 to-white">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            <h3 className="text-lg md:text-xl font-semibold text-gray-900">Answer</h3>
          </div>
          {processingTime && (
            <span className="text-xs md:text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full font-medium">
              {processingTime.toFixed(1)}s
            </span>
          )}
        </div>
      </div>
      <div className="px-6 md:px-8 py-6 md:py-8">
        <div
          ref={contentRef}
          className="prose prose-lg max-w-none 
            prose-headings:text-gray-900 prose-headings:font-semibold
            prose-p:text-gray-700 prose-p:leading-relaxed
            prose-strong:text-gray-900 prose-strong:font-semibold
            prose-code:text-gray-900 prose-code:bg-gray-100 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-code:font-mono
            prose-pre:bg-gray-50 prose-pre:border prose-pre:border-gray-200 prose-pre:rounded-xl prose-pre:shadow-sm
            prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline
            prose-ul:text-gray-700 prose-ol:text-gray-700
            prose-li:text-gray-700"
        />
      </div>
    </div>
  )
}

