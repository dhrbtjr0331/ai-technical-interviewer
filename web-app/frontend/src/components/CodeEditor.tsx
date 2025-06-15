import React, { useRef, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import * as monaco from 'monaco-editor';

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language?: string;
  theme?: string;
  height?: string;
  options?: monaco.editor.IStandaloneEditorConstructionOptions;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  onChange,
  language = 'python',
  theme = 'vs-dark',
  height = '400px',
  options = {}
}) => {
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null);

  const defaultOptions: monaco.editor.IStandaloneEditorConstructionOptions = {
    selectOnLineNumbers: true,
    automaticLayout: true,
    fontSize: 14,
    fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
    lineNumbers: 'on',
    glyphMargin: false,
    folding: true,
    lineDecorationsWidth: 10,
    lineNumbersMinChars: 3,
    minimap: { enabled: false }, // Disable minimap for cleaner look
    scrollBeyondLastLine: false,
    wordWrap: 'on',
    tabSize: 4,
    insertSpaces: true,
    detectIndentation: false,
    bracketPairColorization: { enabled: true },
    suggest: {
      showKeywords: true,
      showSnippets: true,
      showFunctions: true,
      showMethods: true,
      showVariables: true,
    },
    quickSuggestions: {
      other: true,
      comments: true,
      strings: true,
    },
    parameterHints: { enabled: true },
    hover: { enabled: true },
    contextmenu: true,
    ...options
  };

  const handleEditorDidMount = (editor: monaco.editor.IStandaloneCodeEditor) => {
    editorRef.current = editor;
    
    // Focus the editor
    editor.focus();
    
    // Add custom key bindings
    editor.addAction({
      id: 'run-code',
      label: 'Run Code',
      keybindings: [monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter],
      contextMenuGroupId: 'navigation',
      contextMenuOrder: 1.5,
      run: () => {
        // This will be handled by the parent component
        const runEvent = new CustomEvent('run-code');
        window.dispatchEvent(runEvent);
      }
    });

    // Add execute shortcut
    editor.addAction({
      id: 'execute-code',
      label: 'Execute Code',
      keybindings: [monaco.KeyMod.Shift | monaco.KeyCode.Enter],
      run: () => {
        const executeEvent = new CustomEvent('execute-code');
        window.dispatchEvent(executeEvent);
      }
    });
  };

  const handleEditorChange = (value: string | undefined) => {
    onChange(value || '');
  };

  return (
    <div className="code-editor-container">
      <Editor
        height={height}
        language={language}
        theme={theme}
        value={value}
        options={defaultOptions}
        onMount={handleEditorDidMount}
        onChange={handleEditorChange}
        loading={<div className="editor-loading">Loading editor...</div>}
      />
    </div>
  );
};

export default CodeEditor;