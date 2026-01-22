import { ReactNode } from 'react';
import { CheckCircle2, XCircle, Loader2, AlertCircle } from 'lucide-react';

interface TableCardProps {
  title: string;
  step: number;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  children: ReactNode;
  error?: string;
}

export default function TableCard({ title, step, status, children, error }: TableCardProps) {
  const getStatusIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-5 w-5 text-green-600" />;
      case 'in-progress':
        return <Loader2 className="h-5 w-5 text-blue-600 animate-spin" />;
      case 'error':
        return <XCircle className="h-5 w-5 text-red-600" />;
      default:
        return <AlertCircle className="h-5 w-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'completed':
        return 'border-green-200 bg-green-50';
      case 'in-progress':
        return 'border-blue-200 bg-blue-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      default:
        return 'border-gray-200 bg-white';
    }
  };

  return (
    <div className={`rounded-lg border-2 p-6 ${getStatusColor()}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-blue-100 text-blue-600 font-semibold">
            {step}
          </div>
          <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
        </div>
        {getStatusIcon()}
      </div>
      {error && (
        <div className="mb-4 p-3 bg-red-100 border border-red-300 rounded text-red-700 text-sm">
          {error}
        </div>
      )}
      <div>{children}</div>
    </div>
  );
}
