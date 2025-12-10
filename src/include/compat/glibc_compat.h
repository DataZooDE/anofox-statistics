/**
 * Compatibility header for older glibc versions
 *
 * Defines missing constants that were added in:
 * - Linux kernel 3.4
 * - glibc 2.17
 *
 * This allows building on older systems (e.g., conda environments with old glibc)
 */

#ifndef ANOFOX_GLIBC_COMPAT_H
#define ANOFOX_GLIBC_COMPAT_H

/* madvise constants for memory dump control */
#ifndef MADV_DONTDUMP
#define MADV_DONTDUMP 24
#endif

#ifndef MADV_DODUMP
#define MADV_DODUMP 25
#endif

#endif /* ANOFOX_GLIBC_COMPAT_H */
