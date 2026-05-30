import React from 'react';
import { cva } from 'class-variance-authority';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs){
  return twMerge(clsx(inputs));
}

export const Card = ({className, children}) => (
  <div className={cn('rounded-lg border border-white/10 bg-neutral-900/70 backdrop-blur px-4 py-3 shadow-sm', className)}>
    {children}
  </div>
);

export const Button = ({className, children, variant='default', ...rest}) => {
  const base = 'inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-neutral-950 disabled:opacity-50 disabled:pointer-events-none';
  const variants = {
    default: 'bg-indigo-600 hover:bg-indigo-500 text-white focus:ring-indigo-400',
    ghost: 'bg-transparent hover:bg-white/10 text-neutral-200',
    danger: 'bg-red-600 hover:bg-red-500 text-white focus:ring-red-400'
  };
  return <button className={cn(base, variants[variant]||variants.default, className)} {...rest}>{children}</button>;
};

const badgeVariants = cva(
  'inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium uppercase tracking-wide',
  {
    variants: {
      variant: {
        default: 'border-white/10 bg-neutral-800 text-neutral-200',
        success: 'border-emerald-500/40 bg-emerald-500/15 text-emerald-300',
        warning: 'border-amber-500/40 bg-amber-500/15 text-amber-300',
        danger: 'border-red-500/40 bg-red-500/15 text-red-300',
        info: 'border-sky-500/40 bg-sky-500/15 text-sky-300'
      }
    },
    defaultVariants: { variant: 'default' }
  }
);

export const Badge = ({className, variant, children}) => (
  <span className={cn(badgeVariants({variant}), className)}>{children}</span>
);

export const Separator = ({className}) => <div className={cn('h-px w-full bg-gradient-to-r from-transparent via-white/10 to-transparent my-2', className)} />;

