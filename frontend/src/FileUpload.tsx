
import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, File, Image, Code, X, Eye } from 'lucide-react';

interface FileUploadProps {
  onFileUploaded?: (file: any) => void;
  conversationId?: number;
  sharedConversationId?: number;
}

export const FileUpload: React.FC<FileUploadProps> = ({ 
  onFileUploaded, 
  conversationId, 
  sharedConversationId 
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<any[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      uploadFiles(files);
    }
  };

  const uploadFiles = async (files: File[]) => {
    setIsUploading(true);

    for (const file of files) {
      try {
        const formData = new FormData();
        formData.append('file', file);
        if (conversationId) formData.append('conversation_id', conversationId.toString());
        if (sharedConversationId) formData.append('shared_conversation_id', sharedConversationId.toString());

        const response = await fetch('/api/upload', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
          },
          body: formData
        });

        if (response.ok) {
          const uploadedFile = await response.json();
          setUploadedFiles(prev => [...prev, uploadedFile]);
          onFileUploaded?.(uploadedFile);
        }
      } catch (error) {
        console.error('Upload failed:', error);
      }
    }

    setIsUploading(false);
  };

  const getFileIcon = (fileType: string) => {
    switch (fileType) {
      case 'image': return <Image className="w-5 h-5" />;
      case 'code': return <Code className="w-5 h-5" />;
      default: return <File className="w-5 h-5" />;
    }
  };

  const removeFile = (fileId: number) => {
    setUploadedFiles(prev => prev.filter(f => f.file_id !== fileId));
  };

  return (
    <div className="space-y-4">
      {/* Upload Area */}
      <motion.div
        className={`border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-colors ${
          isDragging 
            ? 'border-blue-400 bg-blue-500/10' 
            : 'border-gray-600 hover:border-gray-500'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
      >
        <input
          ref={fileInputRef}
          type="file"
          multiple
          onChange={handleFileSelect}
          className="hidden"
          accept="image/*,.py,.js,.ts,.html,.css,.cpp,.java,.txt,.md,.pdf"
        />
        
        {isUploading ? (
          <div className="flex items-center justify-center space-x-2">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            >
              <Upload className="w-6 h-6 text-blue-400" />
            </motion.div>
            <span className="text-blue-400">Uploading & Analyzing...</span>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="w-8 h-8 text-gray-400 mx-auto" />
            <p className="text-gray-400">
              Drag & drop files here or <span className="text-blue-400">click to select</span>
            </p>
            <p className="text-xs text-gray-500">
              Images, code files, documents (10MB max)
            </p>
          </div>
        )}
      </motion.div>

      {/* Uploaded Files */}
      {uploadedFiles.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-gray-400">Uploaded Files:</h4>
          {uploadedFiles.map((file) => (
            <motion.div
              key={file.file_id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/10 rounded-lg p-3 flex items-center space-x-3"
            >
              <div className="text-blue-400">
                {getFileIcon(file.file_type)}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {file.filename}
                </p>
                {file.analysis && (
                  <p className="text-xs text-gray-400 truncate">
                    {file.analysis.description || file.analysis.analysis || 'Analysis completed'}
                  </p>
                )}
              </div>
              <button
                onClick={() => removeFile(file.file_id)}
                className="p-1 hover:bg-red-500/20 rounded text-red-400"
              >
                <X className="w-4 h-4" />
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
};
