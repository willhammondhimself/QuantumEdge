#!/bin/bash
# Professional Git workflow setup for QuantumEdge

set -e

echo "🔧 Setting up Git hooks and configuration for QuantumEdge..."

# Set commit message template
git config commit.template scripts/commit_template.txt

# Configure Git for better collaboration
git config pull.rebase true
git config push.default simple
git config core.autocrlf input
git config core.editor "code --wait"  # Use VS Code as editor

# Configure better diff and merge tools
git config diff.algorithm histogram
git config merge.conflictstyle diff3

# Set up useful aliases
git config alias.co checkout
git config alias.br branch
git config alias.ci commit
git config alias.st status
git config alias.unstage 'reset HEAD --'
git config alias.last 'log -1 HEAD'
git config alias.visual '!gitk'
git config alias.graph 'log --oneline --graph --decorate --all'
git config alias.pushf 'push --force-with-lease'

# Professional commit aliases
git config alias.amend 'commit --amend --no-edit'
git config alias.amendm 'commit --amend'
git config alias.fixup 'commit --fixup'
git config alias.squash 'commit --squash'

# Useful workflow aliases
git config alias.wip 'commit -am "WIP: work in progress"'
git config alias.unwip 'reset HEAD~1'
git config alias.sync '!git fetch origin && git rebase origin/main'

# Performance and safety
git config core.preloadindex true
git config core.fscache true
git config gc.auto 256

echo "✅ Git configuration complete!"
echo ""
echo "📋 Available Git aliases:"
echo "  git graph    - Show commit graph"
echo "  git sync     - Sync with main branch"
echo "  git wip      - Quick work-in-progress commit"
echo "  git unwip    - Undo last WIP commit"
echo "  git pushf    - Force push safely"
echo ""
echo "💡 Use 'git commit' to see the commit message template"