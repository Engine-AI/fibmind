export interface Config {
  command: string
  args: string[]
  cwd?: string
  env: Record<string, string>
  toolCallTimeoutMs: number
  owner?: string
  projectId?: string
  workspaceId?: string
  hotOnStart: boolean
  hotGoal: string
  hotTopK: number
  hotBudgetTokens: number
  recall: boolean
  recallTopK: number
  recallBudgetTokens: number
  advise: 'deny' | 'ask' | 'off'
  observe: boolean
  reviewMode: 'approve' | 'auto' | 'off'
  skills: boolean
  skillsMinMaturity: 'candidate' | 'verified' | 'established'
}
