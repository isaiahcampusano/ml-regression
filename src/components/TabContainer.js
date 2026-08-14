export function renderTabs(activeTab) {
  return ['playground', 'learn'].map((tab) => `
    <button class="tab ${tab === activeTab ? 'active' : ''}" data-tab="${tab}" aria-selected="${tab === activeTab}">
      ${tab[0].toUpperCase()}${tab.slice(1)}
    </button>`).join('');
}
