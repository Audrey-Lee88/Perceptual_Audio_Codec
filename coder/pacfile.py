"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2019-2024 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec    # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits  # defines the grouping of MDCT lines into scale factor bands

import numpy as np  # to allow conversion of data blocks to numpy's array object
from helper import *
import sys
MAX16BITS = 32767

import pickle
from audio_codec_huffman import *
from huffman_coder.bitarray_util import *

class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag=b'PAC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag=self.fp.read(4)
        if tag!=self.tag: raise RuntimeError("Tried to read a non-PAC file into a PACFile object")
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines_l, nMDCTLines_s, nMDCTLines_t, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLLLHH',self.fp.read(calcsize('<LHLLLLHH')))
        nBands_l = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines_l=  unpack('<'+str(nBands_l)+'H',self.fp.read(calcsize('<'+str(nBands_l)+'H')))
        sfBands_l=ScaleFactorBands(nLines_l)

        nBands_s = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines_s=  unpack('<'+str(nBands_s)+'H',self.fp.read(calcsize('<'+str(nBands_s)+'H')))
        sfBands_s=ScaleFactorBands(nLines_s)

        nBands_t = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines_t=  unpack('<'+str(nBands_t)+'H',self.fp.read(calcsize('<'+str(nBands_t)+'H')))
        sfBands_t=ScaleFactorBands(nLines_t)

        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines_l = myParams.nSamplesPerBlock = nMDCTLines_l
        myParams.nMDCTLines_s = nMDCTLines_s
        myParams.nMDCTLines_t = nMDCTLines_t
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # add in scale factor band information
        myParams.sfBands_l =sfBands_l
        myParams.sfBands_s=sfBands_s
        myParams.sfBands_t=sfBands_t

        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append( np.zeros(nMDCTLines_l, dtype=float) )
        myParams.overlapAndAdd=overlapAndAdd
        return myParams


    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        
        esc_key = 6060
        # # create instance of huffman audio class
        # huff_aud = huffman_audio(mant_dict)

        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=float))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0] # read it as a little-endian unsigned long
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) ) # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:  raise "Only read a partial block of coded PACFile data"

            # extract the data from the PackedBits object
            # CUSTOM DATA:
            # < now can unpack any custom data passed in the nBytes of data >
            codingParams.currBlock = pb.ReadBits(2) # read in blockType

            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            scaleFactor=[]
            bitAlloc=[]
            if codingParams.currBlock == 0:
                nMDCTLines = codingParams.nMDCTLines_l
            elif codingParams.currBlock == 1:
                nMDCTLines = codingParams.nMDCTLines_t
            elif codingParams.currBlock == 2:
                nMDCTLines = codingParams.nMDCTLines_s
            elif codingParams.currBlock == 3:
                nMDCTLines = codingParams.nMDCTLines_t
            mantissa=np.zeros(nMDCTLines,np.int32)  # start w/ all mantissas zero

            huff_bit = pb.ReadBits(1)
            if huff_bit == 1:
                HuffEncode = True
            else:
                HuffEncode = False
            print(huff_bit,'decode')

            for iBand in range(codingParams.sfBands_l.nBands): # loop over each scale factor band to pack its data
                ba = pb.ReadBits(codingParams.nMantSizeBits)
                if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                bitAlloc.append(ba)  # bit allocation for this band
                scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                if bitAlloc[iBand]:
                    if codingParams.currBlock == 0:
                        sfBands = codingParams.sfBands_l
                    elif codingParams.currBlock == 1:
                        sfBands = codingParams.sfBands_t
                    elif codingParams.currBlock == 2:
                        sfBands = codingParams.sfBands_s
                    elif codingParams.currBlock == 3:
                        sfBands = codingParams.sfBands_t
                    # if bits allocated, extract those mantissas and put in correct location in matnissa array
                    m=np.empty(sfBands.nLines[iBand],np.int32)
                    for j in range(sfBands.nLines[iBand]):     
                        # Defaults to huffman
                        # Unless see decoded "A" from Huffman
                        # Then switch to read bits through Mant way
                        if HuffEncode:
                            bitsToRead = 1
                            read_bits_val = pb.ReadBits(bitsToRead) 
                            while True:
                                codeword =  uint_to_bitarray(read_bits_val, bit_width=bitsToRead)
                                try: 
                                    decoded_block, _ = huff_aud.audio_huffman_decode(codeword) 
                                    break
                                except:
                                    bitsToRead += 1
                                    addBit = pb.ReadBits(1) 
                                    read_bits_val <<= 1
                                    read_bits_val += addBit
                            if decoded_block == esc_key:
                                m[j]=pb.ReadBits(bitAlloc[iBand]) 
                            else: 
                                m[j] = decoded_block
                        else:
                            m[j]=pb.ReadBits(bitAlloc[iBand]) 
                    mantissa[sfBands.lowerLine[iBand]:(sfBands.upperLine[iBand]+1)] = m
            # done unpacking data (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can unpack any custom data passed in the nBytes of data >

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)

            if codingParams.currBlock == 0:
                overlap_bound = codingParams.nMDCTLines_l
            elif codingParams.currBlock == 1:
                overlap_bound = codingParams.nMDCTLines_l
            elif codingParams.currBlock == 2:
                overlap_bound = codingParams.nMDCTLines_s
            elif codingParams.currBlock == 3:
                overlap_bound = codingParams.nMDCTLines_s
            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[:overlap_bound]) ) )  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[overlap_bound:]  # save other half for next pass

        # end loop over channels, return signed-fraction samples for this block
        return data


    def WriteFileHeader(self,codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples%codingParams.nMDCTLines_l:
            codingParams.numSamples += (codingParams.nMDCTLines_l
                        - codingParams.numSamples%codingParams.nMDCTLines_l) # zero padding for partial final PCM block
        
        # write the coded file attributes
        self.fp.write(pack('<LHLLLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLines_l, codingParams.nMDCTLines_s, codingParams.nMDCTLines_t,
            codingParams.nScaleBits, codingParams.nMantSizeBits  ))

        sfBands_l=ScaleFactorBands(AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines_l, codingParams.sampleRate))
        codingParams.sfBands_l =sfBands_l
        self.fp.write(pack('<L',sfBands_l.nBands))
        self.fp.write(pack('<'+str(sfBands_l.nBands)+'H',*(sfBands_l.nLines.tolist()) ))

        sfBands_s=ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines_s, codingParams.sampleRate))
        codingParams.sfBands_s=sfBands_s
        self.fp.write(pack('<L',sfBands_s.nBands))
        self.fp.write(pack('<'+str(sfBands_s.nBands)+'H',*(sfBands_s.nLines.tolist()) ))

        sfBands_t=ScaleFactorBands(AssignMDCTLinesFromFreqLimits(round(codingParams.nMDCTLines_t), codingParams.sampleRate))
        codingParams.sfBands_t=sfBands_t
        self.fp.write(pack('<L',sfBands_t.nBands))
        self.fp.write(pack('<'+str(sfBands_t.nBands)+'H',*(sfBands_t.nLines.tolist()) ))

        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines_l,dtype=float) )
        codingParams.priorBlock = priorBlock

        return
    
    def WriteOneBlock(self, data, codingParams, block_num):
        fullBlockData=[]
        for iCh in range(codingParams.nChannels):
            if codingParams.currBlock == 0:
                block_len = codingParams.nMDCTLines_l
            elif codingParams.currBlock == 1:
                block_len = codingParams.nMDCTLines_s
            elif codingParams.currBlock == 2:
                block_len = codingParams.nMDCTLines_s
            elif codingParams.currBlock == 3:
                block_len = codingParams.nMDCTLines_l
            low_bound = block_num*block_len
            upper_bound = block_num*block_len + block_len
            fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh][low_bound:upper_bound]) ) )
            codingParams.priorBlock[iCh] = data[iCh][low_bound:upper_bound]
        
        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header

        if codingParams.currBlock == 0:
            sfBands = codingParams.sfBands_l
        elif codingParams.currBlock == 1:
            sfBands = codingParams.sfBands_t
        elif codingParams.currBlock == 2:
            sfBands = codingParams.sfBands_s
        elif codingParams.currBlock == 3:
            sfBands = codingParams.sfBands_t
        
        # for each channel, write the data to the output file
        # open dictionary for mantissa dict
        esc_key = 6060

        enc_A = huff_aud.audio_huffman_encode(esc_key)
        len_bit_A = len(enc_A)
        enc_A_val = bitarray_to_uint(enc_A)
        
        for iCh in range(codingParams.nChannels):
            nByte_huff = 0
            nByte_norm = 0
            # determine the size of this channel's data block and write it to the output file
            nByte_huff +=codingParams.nScaleBits  # bits for overall scale factor
            nByte_norm +=codingParams.nScaleBits
            iMant=0
            for iBand in range(sfBands.nBands): # loop over each scale factor band to get its bits
                nByte_huff += codingParams.nMantSizeBits+codingParams.nScaleBits
                nByte_norm += codingParams.nMantSizeBits+codingParams.nScaleBits
                # nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                if bitAlloc[iCh][iBand]:
                    # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                    for j in range(sfBands.nLines[iBand]):
                        if mantissa[iCh][iMant+j] in mant_dict:
                            # perform Huffman
                            encoded_sym = huff_aud.audio_huffman_encode(mantissa[iCh][iMant+j])
                            len_bit_arr = len(encoded_sym)
                            if len_bit_arr >= bitAlloc[iCh][iBand]:
                                nByte_huff += bitAlloc[iCh][iBand]
                                nByte_huff += len_bit_A
                            else:
                                nByte_huff += len_bit_arr
                        else:
                            nByte_huff += bitAlloc[iCh][iBand]#*codingParams.sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
                            nByte_huff += len_bit_A
                        nByte_norm += bitAlloc[iCh][iBand]
                    iMant += sfBands.nLines[iBand]
            # end computing bits needed for this channel's data

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>
            
            
            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nByte_norm > nByte_huff:
                nBytes = nByte_huff
                HuffEncode = True
            else:
                nBytes = nByte_norm
                HuffEncode = False
            nBytes += 1    # For huff encode
            nBytes += 2
            print(nBytes)
            if nBytes%BYTESIZE==0:  nBytes //= BYTESIZE
            else: nBytes = nBytes//BYTESIZE + 1
            self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # now pack the nBytes of data into the PackedBits object
            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>
            pb.WriteBits(codingParams.currBlock, 2)

            pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
            iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)

            
            # TODO: ENCODE HUFF OR NOT
            huff_bit = 0
            if HuffEncode:
                huff_bit = 1
            print(huff_bit,'encode')
            pb.WriteBits(huff_bit, 1)
            for iBand in range(sfBands.nBands): # loop over each scale factor band to pack its data
                ba = bitAlloc[iCh][iBand]
                if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                if bitAlloc[iCh][iBand]:
                    for j in range(sfBands.nLines[iBand]):
                        # if j is in our dict, we write bits with Huffman 
                        if HuffEncode:
                            if int(mantissa[iCh][iMant+j]) in mant_dict:
                                # perform Huffman
                                
                                encoded_sym = huff_aud.audio_huffman_encode(int(mantissa[iCh][iMant+j]))
                                len_bit_arr = len(encoded_sym)
                                # check if bits with Huffman < bitAlloc
                                if len_bit_arr >= bitAlloc[iCh][iBand]:
                                    pb.WriteBits(enc_A_val,len_bit_A)  # esc
                                    pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand]) 
                                else:
                                    sym_int = bitarray_to_uint(encoded_sym)
                                    pb.WriteBits(sym_int, len_bit_arr)
                            else:
                                pb.WriteBits(enc_A_val,len_bit_A)  # esc
                                pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                        else:
                            pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])
                    iMant += sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
            # done packing (end loop over scale factor bands)

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return


    def WriteDataBlock(self,data, codingParams, ending_flag = False):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # We want to do 8 sub blocks if we are in a short block one:
        if codingParams.currBlock == 0:
            subBlock = False
        elif codingParams.currBlock == 1:
            subBlock = True
        elif codingParams.currBlock == 2:
            subBlock = True
        elif codingParams.currBlock == 3:
            subBlock = False
        # print("Short Block: ", subBlock)
            
        if subBlock == True and ending_flag == False:
            repeats = 8
            for block_num in range(repeats):
                self.WriteOneBlock(data, codingParams, block_num)
                codingParams.prevBlock = codingParams.currBlock
                if codingParams.prevBlock == 1:
                    codingParams.currBlock = 2
                elif codingParams.prevBlock == 3:
                    codingParams.currBlock = 0
        else:
            self.WriteOneBlock(data, codingParams, 0)
            codingParams.prevBlock = codingParams.currBlock
            if codingParams.prevBlock == 1:
                codingParams.currBlock = 2
            elif codingParams.prevBlock == 3:
                codingParams.currBlock = 0
            
        return        

    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block

            if codingParams.currBlock == 0:
                nMDCTLines = codingParams.nMDCTLines_l
            elif codingParams.currBlock == 1:
                nMDCTLines = codingParams.nMDCTLines_t
            elif codingParams.currBlock == 2:
                nMDCTLines = codingParams.nMDCTLines_t
            elif codingParams.currBlock == 3:
                nMDCTLines = codingParams.nMDCTLines_t

            data = [ np.zeros(nMDCTLines,dtype=float),
                     np.zeros(nMDCTLines,dtype=float) ]
            # print("Writing Zeros Shouldn't Be An Issue")
            # print("Block Types: ", codingParams.currBlock)
            self.WriteDataBlock(data, codingParams, ending_flag = True)
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams)

#-----------------------------------------------------------------------------

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__=="__main__":
    # Change these variables, data rate in Kbps
    # audio_names = ["castanets", "glockenspiel", "harpsichord", "quartet", "spfe", "spgm"]
    audio_names = ["spfe"]
    data_rates = [96]
    # data_rates = [96, 128]

    # Programmatically generate the file name
    for data_rate in data_rates:
        for audio in audio_names:
            mant_dict_file = 'mantissa_dicts/mantissa_dict_for_huff_' + audio + '.pkl'
            file = open(mant_dict_file, 'rb')
            mant_dict = pickle.load(file)
            file.close()
            # create instance of huffman audio class
            huff_aud = huffman_audio(mant_dict)
            input_file = "test/"+audio+"_t.wav"
            output_file = "test/output_wav/"+audio+"_"+str(data_rate)+"kbps.wav"
            pac_file = "test/pacfiles/"+audio+"_"+str(data_rate)+"kbps.pac"

            import time
            from pcmfile import * # to get access to WAV file handling
            elapsed = time.time()

            for Direction in ("Encode", "Decode"):

                # create the audio file objects
                if Direction == "Encode":
                    print( "\n\tEncoding input PCM file... ", input_file, " at ", data_rate)
                    inFile= PCMFile(input_file)
                    outFile = PACFile(pac_file)
                else: # "Decode"
                    print( "\n\tDecoding coded PAC file...", output_file, " at ", data_rate)
                    inFile = PACFile(pac_file)
                    outFile= PCMFile(output_file)
                # only difference is file names and type of AudioFile object

                # open input file
                codingParams=inFile.OpenForReading()  # (includes reading header)

                # pass parameters to the output file
                if Direction == "Encode":
                    # set additional parameters that are needed for PAC file
                    # (beyond those set by the PCM file on open)
                    
                    # Set the block lengths for long, short, and transition blocks (AC-2A)
                    codingParams.nMDCTLines_l = 2048
                    codingParams.nMDCTLines_s = 256
                    codingParams.nMDCTLines_t = int((codingParams.nMDCTLines_l + codingParams.nMDCTLines_s)//2)

                    # Remember the type of block these blocks were
                    # There are four type of blocks: "long", "lToS", "short", "sToL"
                    # We can treat them in this order as an enum.
                    # {"long": 0, "lToS": 1, "short": 2, "sToL": 3}
                    codingParams.currBlock = 0
                    codingParams.prevBlock = 0
                    # We plan to write the previous block and just need to keep track of
                    # whether or not the next block is a transient or not
                    codingParams.nextBlock = False

                    codingParams.nScaleBits = 3
                    codingParams.nMantSizeBits = 4

                    # Automate this based on the data rate provided above
                    codingParams.targetBitsPerSample = (data_rate*1000)/codingParams.sampleRate
                    
                    # tell the PCM file how large the block size is
                    codingParams.nSamplesPerBlock = codingParams.nMDCTLines_l
                else: # "Decode"
                    # set PCM parameters (the rest is same as set by PAC file on open)
                    codingParams.bitsPerSample = 16
                # only difference is in setting up the output file parameters

                # Read the input file and pass its data to the output file to be written
                # open the output file
                outFile.OpenForWriting(codingParams) # (includes writing header)
                

                # Read the input file and pass its data to the output file to be written
                prevBlock = [] 
                currBlock = []
                nextBlock = []

                # Now, you want to read in the first block to get the train started
                currBlock = inFile.ReadDataBlock(codingParams)
                
                # The logic is:
                #   (1) Pull the next block
                #   (2) At the end of every loop, push each block back and "free up" next block
                #   (3) Check the current block at the beginning of every loop to see if the file is over.
                block_num = 0
                while True:
                    
                    if not currBlock: 
                        # print("No Curr Block")
                        break  # we hit the end of the input file

                    # Read in the "next" block too for the lookahead
                    nextBlock = inFile.ReadDataBlock(codingParams)

                    if Direction == "Decode":
                        prevBlock = currBlock
                        currBlock = nextBlock
                        outFile.WriteDataBlock(prevBlock,codingParams)
                        # print( ".",end="")  # just to signal how far we've gotten to user
                        continue 

                    if not nextBlock:
                        # print("No Next Block: ", block_num)
                        prevBlock = currBlock
                        currBlock = nextBlock
                        outFile.WriteDataBlock(prevBlock,codingParams)
                        # print( ".",end="")  # just to signal how far we've gotten to user
                        continue
                    else:
                        # Figure out if the next block is a transient to start a start block
                        for iCh in range(codingParams.nChannels):
                            codingParams.nextBlock = False
                            codingParams.nextBlock = containsTransient(currBlock[iCh], nextBlock[iCh], codingParams.sampleRate)
                            if codingParams.nextBlock == True:
                                # print(codingParams.prevBlock)
                                # print(codingParams.currBlock)
                                # print(codingParams.nextBlock)
                                break
                    
                    if Direction == "Encode":
                        block_num += 1
                        # Work some logic to set the currBlock accurately based on nextBlock
                        if codingParams.nextBlock:
                            # Roll the block states one back
                            codingParams.prevBlock = codingParams.currBlock
                            # if last block was long and next is transient -> l2s
                            if codingParams.prevBlock == 0:
                                codingParams.currBlock = 1
                            # if last block was l2s and next is transient -> short
                            elif codingParams.prevBlock == 1:
                                codingParams.currBlock = 2
                            # if last block was short and next is transient -> short
                            elif codingParams.prevBlock == 2:
                                codingParams.currBlock = 2
                            # if last block was s2l and next is transient -> l2s
                            elif codingParams.prevBlock == 3:
                                codingParams.currBlock = 1
                        else:
                            # Roll the block states one back
                            codingParams.prevBlock = codingParams.currBlock
                            # if last block was long and next is long -> long
                            if codingParams.prevBlock == 0:
                                codingParams.currBlock = 0
                            # if last block was l2s and next is long -> s2l
                            elif codingParams.prevBlock == 1:
                                codingParams.currBlock = 3
                            # if last block was short and next is long -> s2l
                            elif codingParams.prevBlock == 2:
                                codingParams.currBlock = 3
                            # if last block was s2l and next is long -> long
                            elif codingParams.prevBlock == 3:
                                codingParams.currBlock = 0

                        # Update previousBlock, currentBlock
                        prevBlock = currBlock
                        currBlock = nextBlock
                        outFile.WriteDataBlock(prevBlock,codingParams)
                        # print( ".",end="")  # just to signal how far we've gotten to user
                        continue
                # end loop over reading/writing the blocks

                # close the files
                inFile.Close(codingParams)
                outFile.Close(codingParams)
            # end of loop over Encode/Decode

            elapsed = time.time()-elapsed
            print( "\nDone with Encode/Decode test\n")
            print( elapsed ," seconds elapsed")
