// 文档四分类常量（与后端 finance_rag/src/rag/models/document_category.py 保持一致）
export const DOCUMENT_CATEGORIES = [
  { value: 'investment_research', label: '投研类' },
  { value: 'compliance_risk', label: '合规风控类' },
  { value: 'business_operations', label: '业务运营类' },
  { value: 'management', label: '管理类' },
]

// 分类英文值 -> 中文标签；未知或空返回「未分类」
export function categoryLabel(value) {
  if (!value) return '未分类'
  const item = DOCUMENT_CATEGORIES.find(c => c.value === value)
  return item ? item.label : value
}
