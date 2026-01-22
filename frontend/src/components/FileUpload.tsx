import { useRef } from 'react';
import { Upload } from 'lucide-react';

interface FileUploadProps {
  accept?: string;
  onChange: (file: File | null) => void;
  label: string;
  file?: File | null;
  disabled?: boolean;
}

export default function FileUpload({ accept, onChange, label, file, disabled = false }: FileUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (disabled) return;
    const selectedFile = e.target.files?.[0] || null;
    onChange(selectedFile);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">{label}</label>
      <div
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`flex items-center justify-center w-full h-32 border-2 border-dashed rounded-lg transition-colors ${
          disabled 
            ? 'border-gray-200 bg-gray-100 cursor-not-allowed opacity-50' 
            : 'border-gray-300 bg-gray-50 cursor-pointer hover:border-blue-400'
        }`}
      >
        <div className="text-center">
          <Upload className="mx-auto h-8 w-8 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">
            {file ? file.name : 'Click to upload or drag and drop'}
          </p>
          {file && (
            <p className="mt-1 text-xs text-gray-500">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          )}
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept={accept}
          onChange={handleFileChange}
          disabled={disabled}
          className="hidden"
        />
      </div>
    </div>
  );
}
